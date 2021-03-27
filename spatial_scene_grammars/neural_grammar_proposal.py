from collections import namedtuple
import networkx as nx
import numpy as np
import os
import time

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
from spatial_scene_grammars.tree import (
    SceneTree, get_tree_root
)
from spatial_scene_grammars.torch_utils import (
    inv_softplus, inv_sigmoid
)


def estimate_observation_likelihood(candidate_nodes, observed_nodes, gaussian_variance):
    # Dumbest possible version: Chamfer distance of like types, scored according to
    # a gaussian error model of given variance.
    # This is *not* a good observation model, since it doesn't enforce one-to-one
    # correspondence, but I'm using it to get off the ground.
    total_log_prob = 0.
    error_distribution = dist.Normal(0., gaussian_variance)
    for node in candidate_nodes:
        max_ll = -np.inf
        for matching_node in observed_nodes:
            if node.__class__ == matching_node.__class__:
                ll = 0.
                node_vars = node.get_all_continuous_variables_as_vector()
                matching_vars = matching_node.get_all_continuous_variables_as_vector()
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
    def __init__(self, node_prototype, output_size):
        super().__init__()
        self.input_size = node_prototype.get_num_continuous_variables()
        self.output_size = output_size
        if self.input_size > 0:
            hidden_size = 128
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
    ''' Transforms a set of observed nodes to a distribution over
    grammar parameters for the meta-tree, where the grammar parameters
    describe derivations through the meta-tree that are likely to
    generate the observed node. '''
    def __init__(self, meta_tree, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.meta_tree = meta_tree
        self.make_rnn(meta_tree, embedding_size)
        self.make_embedding_modules(meta_tree, embedding_size)

    def make_rnn(self, meta_tree, embedding_size):
        # Makes an RNN that directly regresses grammar parameters
        # defined according to the meta-tree:
        #   For each node in the meta-tree:
        #     - If it is nonterminal, regress a replacement set of product weights
        #       according to the node type (so that any tree generated is still
        #       feasible w.r.t the original tree).
        #     - Regress a (mean-field) mean + variance for each continuous variable
        
        # Each entry in this dictionary maps a node in the meta tree to indices
        # into the RNN output for the product weights, local variable means + vars.
        NodeOutputInds = namedtuple('NodeOutputInds',
            ['product_weight_inds', 'local_variable_means_inds', 'local_variable_vars_inds',
            'derived_variable_means_inds', 'derived_variable_vars_inds']
        )
        self.node_output_info = {}
        curr_output_size = 0
        def add_elems(n):
            nonlocal curr_output_size
            inds = range(curr_output_size, curr_output_size + n)
            curr_output_size += n
            return inds
        for node in meta_tree:
            if isinstance(node, NonTerminalNode):
                if isinstance(node, AndNode):
                    product_weight_inds = []
                elif isinstance(node, OrNode):
                    product_weight_inds = add_elems(len(node.child_types))
                elif isinstance(node, IndependentSetNode):
                    product_weight_inds = add_elems(len(node.child_types))
                elif isinstance(node, GeometricSetNode):
                    product_weight_inds = add_elems(len(node.child_types) + 1)
                else:
                    raise NotImplementedError("Don't know how to encode Nonterminal type %s" % node.__class__.__name__)
            else:
                product_weight_inds = []
            num_derived_variables = sum(np.prod(shape) for shape in node.get_derived_variable_info().values())
            num_local_variables = sum(np.prod(shape) for shape in node.get_local_variable_info().values())
            self.node_output_info[node] = NodeOutputInds(
                product_weight_inds=product_weight_inds,
                derived_variable_means_inds=add_elems(num_derived_variables),
                derived_variable_vars_inds=add_elems(num_derived_variables),
                local_variable_means_inds=add_elems(num_local_variables),
                local_variable_vars_inds=add_elems(num_local_variables)
            )
        
        # Create the actual RNN.
        self.hidden_size = curr_output_size
        self.num_layers = 2
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
        
    def make_embedding_modules(self, meta_tree, embedding_size):
        # Make embedding module for each node type.
        self.node_embeddings_by_type = torch.nn.ModuleDict()
        for node in meta_tree.nodes:
            if node.__class__ not in self.node_embeddings_by_type.keys():
                embedding = NodeEmbedding(node, embedding_size)
                self.node_embeddings_by_type[node.__class__.__name__] = embedding 

    def forward(self, observed_nodes):
        # Initialize RNN
        N_nodes = len(observed_nodes)
        assert self.batch_size == 1
        all_x = torch.empty(N_nodes, self.batch_size, self.embedding_size)
        # Randomize observed node order, since it's a unordered set
        # and we'd like to be robust to that.
        shuffled_nodes = [observed_nodes[k] for k in torch.randperm(N_nodes)]
        for k, node in enumerate(shuffled_nodes):
            attr = node.get_all_continuous_variables_as_vector()
            all_x[k, :, :] = self.node_embeddings_by_type[node.__class__.__name__](attr)
        # Pass through RNN.
        output, _ = self.rnn(all_x, self.hidden_init)
        output = self.final_fc(output)
        # Return the final hidden state, removing the batch dim.
        return output[-1, 0, :]
    
    def get_grammar_parameters_from_actual_tree(self, meta_tree, observed_tree, assign_var=0.01):
        # Descend down the observed tree from the top, figuring out how it
        # fits into the meta-tree. Once we have correspondences, we can
        # apply appropriate inverse-transforms to get the corresponding
        # grammar parameters that would have led (deterministically) to the
        # observed tree.
        meta_root = get_tree_root(meta_tree)
        observed_root = get_tree_root(observed_tree)
        meta_node_to_observed_node_mapping = {}
        unexpanded_nodes = [(meta_root, observed_root)]
        
        x_reconstructed = torch.zeros(self.hidden_size)
        x_reconstructed[:] = np.nan
        def reconstruct_hidden_variables(meta_node, observed_node):
            nonlocal x_reconstructed
            all_inds = self.node_output_info[meta_node]
            derived_attr = observed_node.get_derived_variables_as_vector()
            if len(derived_attr) > 0:
                x_reconstructed[all_inds.derived_variable_means_inds] = derived_attr
                x_reconstructed[all_inds.derived_variable_vars_inds] = inv_softplus(assign_var)
            local_attr = observed_node.get_local_variables_as_vector()
            if len(local_attr):
                x_reconstructed[all_inds.local_variable_means_inds] = local_attr
                x_reconstructed[all_inds.local_variable_vars_inds] = inv_softplus(assign_var)
        def reconstruct_hidden_choices(meta_node, observed_node, child_inds):
            # Hidden choices is a list of the indices of the active children
            # of meta_node
            nonlocal x_reconstructed
            all_inds = self.node_output_info[meta_node]    
            x_reconstructed[all_inds.product_weight_inds] = 0.
            assert isinstance(observed_node, NonTerminalNode)
            if isinstance(observed_node, AndNode):
                pass # No choices, no appearance in x_reconstructed
            elif isinstance(observed_node, OrNode):
                # Invert softmax -- make everything big
                # TODO this is ugly. Major refactor where the post-transforms
                # of each variable set (means / vars / etc) are tracked more sensibly
                # are necessary if this starts panning out.
                x_reconstructed[all_inds.product_weight_inds] = -10
                x_reconstructed[all_inds.product_weight_inds[child_inds]] = 10.
            elif isinstance(observed_node, IndependentSetNode):
                # Invert sigmoid via logit.
                desired = torch.zeros(len(all_inds.product_weight_inds))
                desired[child_inds] = 1.
                x_reconstructed[all_inds.product_weight_inds] = inv_sigmoid(desired)
            elif isinstance(observed_node, GeometricSetNode):
                # Invert Softmax -- make everything big. But remember that the encoding
                # is a 1-hot encoding of the number of children to have, so convert
                # from child inds to that.
                num_children = len(child_inds)
                x_reconstructed[all_inds.product_weight_inds] = -10.
                # Tensor conversion so we can use the child_inds tensor to index
                # into the weight inds.
                x_reconstructed[all_inds.product_weight_inds[num_children]] = 10.
            else:
                raise NotImplementedError("Node type ", observed_node.__class__.__name__)

        while len(unexpanded_nodes) > 0:
            meta_node, observed_node = unexpanded_nodes.pop(0)
            reconstruct_hidden_variables(meta_node, observed_node)
            if isinstance(observed_node, NonTerminalNode):
                observed_children = list(observed_tree.successors(observed_node))
                meta_children = list(meta_tree.successors(meta_node))
                child_inds = observed_node.get_child_indices_into_maximal_child_list(observed_children)
                reconstruct_hidden_choices(meta_node, observed_node, child_inds)
                # Not sure I can trust nx node ordering? If these asserts are true,
                # ordering is safe.
                for k, child in enumerate(meta_children):
                    assert isinstance(child, observed_node.get_maximal_child_type_list()[k])
                for child_ind, observed_child in zip(child_inds, observed_children):
                    assert type(meta_children[child_ind]) == type(observed_child)
                    unexpanded_nodes.append((meta_children[child_ind], observed_child))
        return x_reconstructed

    def sample_tree_from_grammar_vector(self, meta_tree, x):
        assert len(x.shape) == 1 and x.shape[0] == self.hidden_size
        # This is largely disgusting to have all in one method and ought
        # to be supported first-class by the scene tree. But I'm prototyping...
        
        def sample_children_from_meta_tree_node(meta_node, new_node):
            # Samples a child set, creates the new nodes, and calls
            # the conditioned sample child function on new_node.
            inds = self.node_output_info[meta_node].product_weight_inds
            raw_product_weights = x[inds]
            child_candidates = list(meta_tree.successors(meta_node))
            assert isinstance(meta_node, NonTerminalNode)
            if isinstance(meta_node, AndNode):
                # The complete list of successors is correct.
                children = child_candidates
            elif isinstance(meta_node, OrNode):
                # Choose one from the list of successors.
                product_weights = torch.nn.functional.softmax(raw_product_weights, dim=0)
                child_ind = pyro.sample("decode_children_sample",
                                        dist.Categorical(product_weights))
                children = [child_candidates[child_ind]]
            elif isinstance(meta_node, IndependentSetNode):
                product_weights = torch.sigmoid(raw_product_weights)
                # Independent choose whether to include each.
                child_inclusion = pyro.sample("decode_children_sample",
                                              dist.Bernoulli(product_weights))
                children = [child_candidates[k] for k, value in enumerate(child_inclusion) if value]
            elif isinstance(meta_node, GeometricSetNode):
                # Choose count
                product_weights = torch.nn.functional.softmax(raw_product_weights, dim=0)
                num_children = pyro.sample("decode_children_sample",
                                           dist.Categorical(product_weights))
                print("Sampled num children: ", num_children)
                children = child_candidates[:num_children]
            else:
                raise NotImplementedError("Don't know how to decode Nonterminal type %s" % meta_node.__class__.__name__)
                
            new_children = [type(meta_child)() for meta_child in children]
            with pyro.poutine.block():
                new_node.conditioned_sample_children(new_children)
            return children, new_children

        def instantiate_node(meta_node, new_node):
            # Pull out the distribution parameters for the local vars, and
            # sample them.
            all_inds = self.node_output_info[meta_node]
            
            def pack_dict(dict_of_shapes, z):
                # Dangerously assumes deterministic orders...
                # The complete mapping per-variable-name for each node of the
                # meta-tree into the encoded grammar vector could conceivably
                # be generated and stored rather than derived-and-local-vectorized
                # blocks?
                out_dict = {}
                k = 0
                for key, shape in dict_of_shapes.items():
                    k_this = np.prod(shape)
                    out_dict[key] = z[k:(k + k_this)].reshape(shape)
                    k += k_this
                return out_dict
        
            derived_means = x[all_inds.derived_variable_means_inds]
            derived_vars = torch.nn.functional.softplus(x[all_inds.derived_variable_vars_inds])
            # Reparam to ensure we can optimize directly without REINFORCE
            derived_attrs = derived_means + derived_vars * pyro.sample("decode_derived_attrs_sample",
                                        dist.Normal(torch.zeros(derived_vars.shape),
                                                    torch.ones(derived_vars.shape)))
            derived_attrs = pack_dict(meta_node.get_derived_variable_info(), derived_attrs)
            local_means = x[all_inds.local_variable_means_inds]
            local_vars = torch.nn.functional.softplus(x[all_inds.local_variable_vars_inds])
            local_attrs = local_means = local_vars * pyro.sample("decode_local_attrs_sample",
                                      dist.Normal(torch.zeros(local_means.shape),
                                                  torch.ones(local_means.shape)))
            local_attrs = pack_dict(meta_node.get_local_variable_info(), local_attrs)
            
            with pyro.poutine.block():
                new_node.conditioned_instantiate(derived_attrs, local_attrs)
            
        def make_new_tree():
            # First choose tree topology by traversing meta-tree.
            # We're traversing the meta-tree, but building our own
            # copy of the set of nodes we've traversed at the same time.
            # This is nasty...
            meta_root_node = get_tree_root(meta_tree)
            new_tree = SceneTree()
            new_root_node = type(meta_root_node)()
            new_tree.add_node(new_root_node)
            # Elements are tuple (meta_tree_node, new_tree_node)
            expansion_queue = [(meta_root_node, new_root_node)]
            while len(expansion_queue) > 0:
                meta_expand_node, new_expand_node = expansion_queue.pop(0)
                # Instantiate this node, using the node in the meta tree to
                # look up the appropriate parts from the grammar vector.
                with scope(prefix=meta_expand_node.name):
                    instantiate_node(meta_expand_node, new_expand_node)
                    if isinstance(meta_expand_node, NonTerminalNode):
                        # Selects a child set from the meta tree node using the
                        # passed grammar vector as selection weights.
                        meta_children, new_children = sample_children_from_meta_tree_node(
                            meta_expand_node, new_expand_node)
                        # Create copies of children of the same types in our tree.
                        for meta_child, new_child in zip(meta_children, new_children):
                            new_tree.add_node(new_child)
                            new_tree.add_edge(new_expand_node, new_child)
                            expansion_queue.append((meta_child, new_child))
            return new_tree
        proposal_trace = pyro.poutine.trace(make_new_tree).get_trace()
        # Compute log prob for all + for just non-reparam'd nodes.
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

    def get_product_weights_and_inclusion_lls(self, meta_tree, x):
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

    def get_variable_distributions_for_meta_node(self, meta_node, x):
        assert meta_node in self.meta_tree.nodes

        all_inds = self.node_output_info[meta_node]

        derived_means = x[all_inds.derived_variable_means_inds]
        derived_vars = torch.nn.functional.softplus(x[all_inds.derived_variable_vars_inds])
        derived_attr_dist = dist.Normal(derived_means, derived_vars)
            
        local_means = x[all_inds.local_variable_means_inds]
        local_vars = torch.nn.functional.softplus(x[all_inds.local_variable_vars_inds])
        local_attr_dist = dist.Normal(local_means, local_vars)

        return derived_attr_dist, local_attr_dist