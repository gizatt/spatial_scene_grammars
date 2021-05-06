from collections import namedtuple
from copy import deepcopy
import networkx as nx
import numpy as np
import os
import time
from typing import Dict

import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope

from spatial_scene_grammars.nodes import (
    NonTerminalNode,
    OrNode,
    GeometricSetNode,
    IndependentSetNode,
    AndNode
)
from spatial_scene_grammars.scene_grammar import (
    SceneGrammarBase,
    SceneTree,
    get_tree_root
)
from spatial_scene_grammars.torch_utils import (
    inv_softplus, inv_sigmoid, ConstrainedParameter
)

# Based on TORCH.NN.UTILS.CONVERT_PARAMETERS
def dict_to_vector(prototype_dict: Dict[str, torch.Tensor]):
    vec = []
    for key, value in prototype_dict.items():
        vec.append(value.view(-1))
    return torch.cat(vec)

def vector_to_dict_of_constrained_params(
            vec: torch.Tensor,
            prototype_dict: Dict[str, ConstrainedParameter]
        ) -> Dict[str, ConstrainedParameter]:
    # Pointer for slicing the vector for each parameter
    pointer = 0
    out_dict = {}
    for key, value in prototype_dict.items():
        # Get length of underlying value.
        tensor_val = value()
        numel = tensor_val.numel()
        # Make a new ConstrainedParameter as a copy of the prototype.
        new_cparam = deepcopy(value)
        # Slice the vector, reshape it, and replace the old data of the parameter
        new_cparam.set_unconstrained(vec[pointer:pointer + numel].view_as(tensor_val))
        out_dict[key] = new_cparam
        # Increment the pointer
        pointer += numel
    return out_dict

def estimate_observation_likelihood(candidate_nodes, observed_nodes, gaussian_variance,
                                    detach_first=False, detach_second=False):
    # Score the distance between two node sets using a Gaussian penalty on
    # node-wise Chamfer distance between variables of same-type nodes.
    # Not an ideal observation model, since it doesn't enforce one-to-one
    # correspondence, but it's not totally baseless.
    total_log_prob = torch.tensor([0.])
    error_distribution = dist.Normal(0., gaussian_variance)
    for node in candidate_nodes:
        max_ll = -np.inf
        for matching_node in observed_nodes:
            if node.__class__ == matching_node.__class__:
                ll = 0.
                node_vars = node.get_all_continuous_variables_as_vector()
                matching_vars = matching_node.get_all_continuous_variables_as_vector()
                if detach_first:
                    node_vars = node_vars.detach()
                if detach_second:
                    matching_vars = matching_vars.detach()
                distances = node_vars - matching_vars
                ll += error_distribution.log_prob(distances).sum()
                if ll > max_ll:
                    max_ll = ll
        total_log_prob += max_ll
    return total_log_prob / max(len(candidate_nodes), 1)


class NodeEmbedding(torch.nn.Module):
    ''' Takes a node class (instantiated, but it doesn't matter with what)
        and prepares an embedding module for it that will transform the
        set of local variables of the node into a fixed size output. '''
    def __init__(self, node_type, output_size, hidden_size=128):
        super().__init__()
        node_prototype = node_type.init_with_default_parameters()
        self.input_size = node_prototype.get_num_continuous_variables()
        self.output_size = output_size
        if self.input_size > 0:
            self.fc1 = torch.nn.Linear(self.input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, self.output_size)
        else:
            self.output_vec = torch.nn.Parameter(
                torch.normal(mean=0., std=1., size=self.output_size)
            )
    def forward(self, x):
        if self.input_size > 0:
            x = self.fc1(x)
            x = x.relu()
            x = self.fc2(x)
            return x
        else:
            return self.output_vec


class GrammarEncoder(torch.nn.Module):
    ''' Encodes a set of observed nodes to a distribution over
    grammar parameters for a specified grammar that might produce
    those nodes, where the grammar parameters describe derivations
    through that grammar that are likely to generate the observed nodes.
    (In other words, this is an inverse procedural model using the
    specified grammar as the model and the observed node set as the
    target output.) '''
    def __init__(self, inference_grammar, embedding_size):
        super().__init__()
        self.inference_grammar = inference_grammar
        self.embedding_size = embedding_size
        self.make_rnn(inference_grammar, embedding_size)
        self.make_embedding_modules(inference_grammar, embedding_size)

    def make_rnn(self, inference_grammar, embedding_size):
        # Makes an RNN that directly regresses grammar parameters of the
        # inference grammar.
        assert isinstance(inference_grammar, SceneGrammarBase)
        grammar_params = inference_grammar.get_default_param_dict()
        # Resolve them to tensors so we can compute size.
        grammar_params_constrained = {key: value() for key, value in grammar_params.items()}
        x = dict_to_vector(grammar_params_constrained)
        n_parameters = len(x)

        # Save the canonical grammar parameter dict as a size reference.
        self.canonical_grammar_param_dict = grammar_params_constrained

        # Create the actual RNN.
        self.hidden_size = n_parameters
        self.num_layers = 3
        self.batch_size = 1
        self.rnn = torch.nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        # And an FC layer on the end to allow arbitrary regression.
        self.final_fc = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_init = torch.nn.Parameter(
            torch.normal(mean=0., std=1., size=(self.num_layers, self.batch_size, self.hidden_size))
        )
        
    def make_embedding_modules(self, inference_grammar, embedding_size):
        # Make embedding module for each node type.
        self.node_embeddings_by_type = torch.nn.ModuleDict()
        for node_type in inference_grammar.get_all_types_in_grammar():
            if node_type not in self.node_embeddings_by_type.keys():
                embedding = NodeEmbedding(node_type, embedding_size)
                self.node_embeddings_by_type[node_type.__name__] = embedding 

    def forward(self, observed_nodes, detach=True):
        # Detach: Detach node attributes.
        # Initialize RNN
        N_nodes = len(observed_nodes)
        assert self.batch_size == 1
        all_x = torch.empty(N_nodes, self.batch_size, self.embedding_size)
        # Randomize observed node order, since it's a unordered set
        # and we'd like to be robust to that.
        shuffled_nodes = [observed_nodes[k] for k in torch.randperm(N_nodes)]
        for k, node in enumerate(shuffled_nodes):
            attr = node.get_all_continuous_variables_as_vector()
            if detach:
                attr = attr.detach()
            all_x[k, :, :] = self.node_embeddings_by_type[node.__class__.__name__](attr)
        # Pass through RNN.
        if N_nodes > 0:
            output, _ = self.rnn(all_x, self.hidden_init)
        else:
            output = self.hidden_init
        output = self.final_fc(output)
        # Return the final hidden state, removing the batch dim.
        return output[-1, 0, :]

    def sample_tree_from_grammar_vector(self, x):
        # Set our inference grammar's parameters accordingly.
        assert len(x.shape) == 1 and x.shape[0] == self.hidden_size
        params = vector_to_dict_of_constrained_params(x, self.inference_grammar.get_default_param_dict())
        # Get tree, and also its proposal density (including only the
        # non-reparam'd part).
        proposal_trace = pyro.poutine.trace(self.inference_grammar.forward).get_trace(params)
        proposal_trace.compute_log_prob()

        total_ll = 0.
        total_nonreparam_ll = 0.

        for key, value in proposal_trace.nodes.items():
            if value["type"] == "sample":
                total_ll += value["log_prob_sum"]
                # TODO: Scaling / other pyro fanciness not handled.
                if not value["fn"].has_rsample:
                    total_nonreparam_ll += value["log_prob_sum"]

        assert torch.isclose(total_ll, proposal_trace.log_prob_sum())
        return proposal_trace.nodes["_RETURN"]["value"], total_ll, total_nonreparam_ll

    def score_tree_with_grammar_vector(self, tree, x):
        assert len(x.shape) == 1 and x.shape[0] == self.hidden_size
        params = vector_to_dict_of_constrained_params(x, self.inference_grammar.get_default_param_dict())
        # Get tree, and also its proposal density (including only the
        # non-reparam'd part).
        return self.inference_grammar.score(tree, params=params)

    def get_product_weights_and_inclusion_lls(self, meta_tree, x):
        # TODO: Should this be a grammar function?
        # Top-down, calculate product weights and production inclusion probalities
        # for the potential children.
        starting_node = get_tree_root(meta_tree)
        inclusion_log_likelihood_per_node = {starting_node: 0.}
        product_weights_per_node = {}
        node_queue = [starting_node]
        while len(node_queue) > 0:
            meta_node = node_queue.pop(0)
            parent_inclusion_ll = inclusion_log_likelihood_per_node[meta_node]
            # Calculate child product weights and corresponding inclusion
            # probs per-child.
            if isinstance(meta_node, NonTerminalNode):
                potential_children = list(meta_tree.successors(meta_node))
                inds = self.node_output_info[meta_node].product_weight_inds
                raw_product_weights = x[inds]
                product_weights = None
                if isinstance(meta_node, AndNode):
                    inclusion_lls = torch.zeros(len(potential_children))
                elif isinstance(meta_node, OrNode):
                    product_weights = torch.nn.functional.softmax(raw_product_weights, dim=0)
                    inclusion_lls = torch.log(product_weights)
                elif isinstance(meta_node, IndependentSetNode):
                    product_weights = torch.sigmoid(raw_product_weights)
                    inclusion_lls = torch.log(product_weights)
                elif isinstance(meta_node, GeometricSetNode):
                    # Choose count
                    product_weights = torch.nn.functional.softmax(raw_product_weights, dim=0)
                    inclusion_lls = torch.log(torch.cumsum(product_weights))
                else:
                    raise NotImplementedError("Don't know how to decode Nonterminal type %s" % meta_node.__class__.__name__)
                product_weights_per_node[meta_node] = product_weights
                
                inclusion_lls = parent_inclusion_ll + inclusion_lls
                for k, meta_child_node in enumerate(potential_children):
                    # Set meta child node appearance log-prob.
                    inclusion_log_likelihood_per_node[meta_child_node] = inclusion_lls[k]
                    node_queue.append(meta_child_node)
        return inclusion_log_likelihood_per_node, product_weights_per_node
