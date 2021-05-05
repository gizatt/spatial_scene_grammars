from copy import deepcopy
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from functools import partial
import networkx as nx

import pyro
from pyro.contrib.autoname import scope, name_count
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints

from .torch_utils import ConstrainedParameter
from .nodes import (
    NonTerminalNode, TerminalNode, Node,
    AndNode, OrNode, GeometricSetNode, IndependentSetNode
)
from .rules import ProductionRule
from .scene_generative_program import SceneGenerativeProgram
from .distributions import LeftSidedRepeatingOnesDist


def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node


class SceneGrammarBase(SceneGenerativeProgram):
    '''
    Basic form of a grammar derived from nodes from nodes.py
    '''
    def __init__(self, root_node_type, root_node_instantiation_dict):
        super().__init__()
        self.root_node_type = root_node_type
        self.root_node_instantiation_dict = root_node_instantiation_dict

    def get_all_types_in_grammar(self):
        return self.get_all_types_in_grammar_given_root(self.root_node_type)

    @staticmethod
    def get_all_types_in_grammar_given_root(root_node_type):
        all_types = set()
        input_queue = [root_node_type]
        while len(input_queue) > 0:
            curr_type = input_queue.pop(0)
            all_types.add(curr_type)
            if issubclass(curr_type, NonTerminalNode):
                for new_type in curr_type.init_with_default_parameters().get_maximal_child_type_list():
                    if new_type not in all_types:
                        input_queue.append(new_type)
        return all_types

class SceneGrammar(SceneGrammarBase):
    '''
    Manages a scene grammar that produces scene trees by composition
    of subclasses of the node types in this repo.
    '''

    def __init__(self, root_node_type, root_node_instantiation_dict, do_sanity_checks=True):
        ''' Given a root node type and an instantiation dict specifying its
        derived variable distributions, prepares this grammar for use. '''
        super().__init__(root_node_type=root_node_type,
                         root_node_instantiation_dict=root_node_instantiation_dict)
        self.root_node_type = root_node_type
        self.root_node_instantiation_dict = root_node_instantiation_dict
        self.do_sanity_checks = do_sanity_checks
        # Build our set of parameters that contains every node's parameter set.
        self.default_params = {}
        for node_type in self.get_all_types_in_grammar():
            params = node_type.get_default_parameters()
            for name, param in params.items():
                full_name = self._get_name_for_param(node_type, name)
                self.default_params[full_name] = param

    @staticmethod
    def _get_name_for_param(node_type, param_name):
        return "%s:%s" % (node_type.__name__, param_name)

    def get_default_param_dict(self):
        return self.default_params

    def get_params_for_node_type(self, node_type, params):
        out_params = {}
        for name in node_type.get_default_parameters().keys():
            full_name = self._get_name_for_param(node_type, name)
            out_params[name] = params[full_name]
        return out_params

    def _spawn_node_with_our_params(self, node_type, params):
        # Dict of ConstrainedParams for this node type
        params_for_node = self.get_params_for_node_type(node_type, params)
        return node_type(parameters=params_for_node)

    def _generate_from_node_recursive(self, scene_tree, parent_node, params):
        if isinstance(parent_node, NonTerminalNode):
            # Choose what gets generated.
            child_types = parent_node.sample_children()
            child_derived_dicts = parent_node.get_derived_variable_dists_for_children(child_types)
            # Spawn and instantiate the children.
            children = [self._spawn_node_with_our_params(child_type, params)
                        for child_type in child_types]
            for child_node, child_dict in zip(children, child_derived_dicts):
                child_node.instantiate(child_dict)
                scene_tree.add_node(child_node)
                scene_tree.add_edge(parent_node, child_node)
                scene_tree = self._generate_from_node_recursive(scene_tree, child_node, params)
        return scene_tree

    def forward(self, params=None):
        # Samples a tree, ensuring our stored parameters get substituted
        # into every node that is generated.
        if params is None:
            params = self.get_default_param_dict()

        scene_tree = SceneTree()
        root_node = self._spawn_node_with_our_params(self.root_node_type, params)
        root_node.instantiate(self.root_node_instantiation_dict)
        scene_tree.add_node(root_node)
        return self._generate_from_node_recursive(scene_tree, root_node, params)

    def _regen_tree_under_new_params(self, scene_tree, params, detach):
        target_root = get_tree_root(scene_tree)
        node_queue = [(target_root, self.root_node_instantiation_dict, None)]
        new_tree = SceneTree()
        while len(node_queue) > 0:
            # Create the clone node.
            target_curr_node, curr_node_inst_dict, parent = node_queue.pop(0)
            resampled_node = self._spawn_node_with_our_params(type(target_curr_node), params)
            new_tree.add_node(resampled_node)
            if parent is not None:
                new_tree.add_edge(parent, resampled_node)
            # Sample its local variables given parent info.
            def maybe_detach_all(dict):
                if detach:
                    return {k: v.detach() for k, v in dict.items()}
                else:
                    return dict
            resampled_node.instantiate(
                curr_node_inst_dict,
                observed_derived_variables=maybe_detach_all(target_curr_node.get_derived_variable_values()),
                observed_local_variables=maybe_detach_all(target_curr_node.get_local_variable_values())
            )
            # Sanity-check that variables came out identical.
            if self.do_sanity_checks:
                assert torch.allclose(resampled_node.get_all_continuous_variables_as_vector(),
                                      target_curr_node.get_all_continuous_variables_as_vector())
            if isinstance(target_curr_node, NonTerminalNode):
                target_children = list(scene_tree.successors(target_curr_node))
                # Simulate sampling the desired child set.
                target_child_types = [type(c) for c in target_children]
                new_child_types = resampled_node.sample_children(observed_child_types=target_child_types)
                if self.do_sanity_checks:
                    # Imperfect sanity-check that the child sets did come out identical.
                    assert len(new_child_types) == len(target_child_types), (new_child_types, target_child_types)
                    for c1, c2 in zip(target_child_types, new_child_types):
                        assert c1 == c2
                # Prepare child derived variable dists.
                child_derived_dicts = resampled_node.get_derived_variable_dists_for_children(new_child_types)
                for target_child, new_child_dict in zip(target_children, child_derived_dicts):
                    node_queue.append((target_child, new_child_dict, resampled_node))
        return new_tree

    def score(self, scene_tree, params=None, detach=False):
        ''' Scores given tree under this grammar using the currently stored
        parameter values and node attributes. '''
        # Regenerate the tree using our local parameter store, conditioning
        # the actual sampling to take the target scene tree values.
        if params is None:
            params = self.get_default_param_dict()
        new_tree = self._regen_tree_under_new_params(scene_tree, params, detach=detach)
        return new_tree.get_log_prob()

    @staticmethod
    def make_meta_scene_tree(root_node_type):
        ''' Given a root node, generates a meta-tree of node types (without
        continuous variables) for which any generated tree from this root is
        a subgraph (again not considering continuous variables). '''
        meta_tree = nx.DiGraph()
        root_node = root_node_type.init_with_default_parameters()
        meta_tree.add_node(root_node)
        node_queue = [root_node]
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            if isinstance(node, TerminalNode):
                continue
            new_node_types = node.get_maximal_child_type_list()
            for new_node_type in new_node_types:
                new_node = new_node_type.init_with_default_parameters()
                meta_tree.add_node(new_node)
                meta_tree.add_edge(node, new_node)
                node_queue.append(new_node)
        return meta_tree


class FullyParameterizedGrammar(SceneGrammarBase):
    '''
    Manages a grammar that follows the same structure as a normal
    SceneGrammar, but instead of following the node's supplied
    generative rules, provides parameterization to control all
    of the choices that can be made in the grammar rollout.

    Used as a representation for parsing.
    '''

    def __init__(self, root_node_type, root_node_instantiation_dict, do_sanity_checks=True):
        ''' Given a root node type and an instantiation dict specifying its
        derived variable distributions, prepares this grammar for use. '''
        super().__init__(root_node_type=root_node_type,
                         root_node_instantiation_dict=root_node_instantiation_dict)
        self.do_sanity_checks = do_sanity_checks

        self.default_params = {}
        # But our database of what the decision making (and corresponding parameters)
        # that is done by each node type.
        for node_type in self.get_all_types_in_grammar():
            self._add_default_params_for_node_type(node_type)

    def get_default_param_dict(self):
        return self.default_params

    def _add_default_params_for_node_type(self, node_type):
        # Sets up the default param entries for this node.
        # If the node is nonterminal, there'll be a param named
        # "node_type.__name__:child_weight".
        # For every attribute of the node, it'll have params
        # named "node_type.__name__:attr_name:[mean, var]".

        # Set up child_weights by looking at the child set of this node.
        node = node_type.init_with_default_parameters()
        if isinstance(node, NonTerminalNode):
            all_child_types = node.get_maximal_child_type_list()
            if isinstance(node, AndNode):
                child_weights = ConstrainedParameter(torch.tensor([]))
            elif isinstance(node, OrNode):
                child_weights = ConstrainedParameter(node.production_weights.detach(),
                                                     constraint=constraints.simplex)
            elif isinstance(node, IndependentSetNode):
                child_weights = ConstrainedParameter(node.production_probs.detach(),
                                                     constraint=constraints.unit_interval)
            elif isinstance(node, GeometricSetNode):
                p = node.geometric_prob
                if isinstance(p, torch.Tensor):
                    p = p.detach()
                # Start with a prior indicating mirroring the original node's chance
                # of sampling each number of children.
                # TODO: Slight error on the probability of picking the final
                # child, since the probability is actually a bit higher due to the
                # truncation.
                n_children_choices = torch.arange(0, len(node.child_types) + 1)
                geometric_child_choice_probs = torch.exp(dist.Geometric(p).log_prob(n_children_choices))
                child_weights = ConstrainedParameter(geometric_child_choice_probs,
                                                     constraint=constraints.simplex)
            else:
                raise NotImplementedError("Don't know how to encode Nonterminal type %s" % node.__class__.__name__)
            full_name = "%s:%s" % (node_type.__name__, "child_weights")
            self.default_params[full_name] = child_weights
        else:
            child_weights = None

        # Create mean field approximations of the continuous variables
        # of this node type.
        def create_mean_field_params(dict_of_variable_infos):
            for key, info in dict_of_variable_infos.items():
                if info.support is not None:
                    raise NotImplementedError("Constrained mean-field estimates not implemented for type ", type(info.support))
                mean = ConstrainedParameter(torch.zeros(info.shape))
                full_name = "%s:%s:%s" % (node_type.__name__, key, "mean")
                self.default_params[full_name] = mean
                var = ConstrainedParameter(torch.ones(info.shape), constraint=constraints.positive)
                full_name = "%s:%s:%s" % (node_type.__name__, key, "var")
                self.default_params[full_name] = var
            
        derived_attr_infos = node_type.get_derived_variable_info()
        create_mean_field_params(derived_attr_infos)
        local_attr_infos = node_type.get_local_variable_info()
        create_mean_field_params(local_attr_infos)

    def _make_variable_mean_fields(self, node_type, params):
        # For a given node type, creates the derived and local
        # variable mean fields.
        # Build distributions for sampling both attr types
        derived_var_dists = {}
        for key, info in node_type.get_derived_variable_info().items():
            mean_name = "%s:%s:%s" % (node_type.__name__, key, "mean")
            var_name = "%s:%s:%s" % (node_type.__name__, key, "var")
            derived_var_dists[key] = dist.Normal(
                params[mean_name](), params[var_name]()
            )
        local_var_dists = {}
        for key, info in node_type.get_local_variable_info().items():
            mean_name = "%s:%s:%s" % (node_type.__name__, key, "mean")
            var_name = "%s:%s:%s" % (node_type.__name__, key, "var")
            local_var_dists[key] = dist.Normal(
                params[mean_name](), params[var_name]()
            )
        return derived_var_dists, local_var_dists

    def _instantiate_node_from_mean_field(self, node, params):
        derived_var_dists, local_var_dists = self._make_variable_mean_fields(type(node), params)
        node.instantiate(derived_variable_distributions=derived_var_dists,
                         local_variable_distributions_override=local_var_dists)

    def _make_inclusion_dist(self, parent_node, params):
        # For a given node type, creates its child inclusion dist
        # from our parameters.
        node_type = type(parent_node)
        assert isinstance(parent_node, NonTerminalNode)
        # Get *constrained* values of the child weights to be used
        # as child-choosing distribution parameters.
        full_name = "%s:%s" % (node_type.__name__, "child_weights")
        child_weights = params[full_name]()
        child_candidates = parent_node.get_maximal_child_type_list()
        # Craft the overriding child inclusion dist by reinterpreting the
        # constrained child weights appropriately for each node type.
        if isinstance(parent_node, AndNode):
            # The complete list of successors is correct.
            inclusion_dist = dist.Bernoulli(torch.ones(len(child_candidates))).to_event(1)
        elif isinstance(parent_node, OrNode):
            # Choose one from the list of successors.
            inclusion_dist = dist.OneHotCategorical(child_weights)
        elif isinstance(parent_node, IndependentSetNode):
            # Independent choose whether to include each.
            inclusion_dist = dist.Bernoulli(child_weights).to_event(1)
        elif isinstance(parent_node, GeometricSetNode):
            # Choose how many children to have and create a vector of
            # 1's up to that many children.
            inclusion_dist = LeftSidedRepeatingOnesDist(child_weights)
        else:
            raise NotImplementedError("Don't know how to decode Nonterminal type %s" % meta_node.__class__.__name__)
        return inclusion_dist
        
    def _sample_children_from_parent_node(self, parent_node, params):
        # Creates an overriding child-selecting distribution for
        # the parent node and samples a set of child types using that
        # overriding distribution.
        inclusion_dist = self._make_inclusion_dist(parent_node, params)
        return parent_node.sample_children(child_inclusion_dist_override=inclusion_dist)

    def _generate_from_node_recursive(self, scene_tree, parent_node, params):
        # Assume parent_node is a fully instantiated node.
        # If it's nonterminal, choose and instantiate its
        # children, and call this on each of them.
        if isinstance(parent_node, NonTerminalNode):
            # Choose what gets generated.
            child_types = self._sample_children_from_parent_node(parent_node, params)

            # Instantiate the node using our mean field params.
            for child_type in child_types:
                # Default params will have no influence on node variables,
                # but this allows us to easily spawn the node.
                child_node = child_type.init_with_default_parameters()
                self._instantiate_node_from_mean_field(child_node, params)
                scene_tree.add_node(child_node)
                scene_tree.add_edge(parent_node, child_node)
                scene_tree = self._generate_from_node_recursive(scene_tree, child_node, params)
        return scene_tree

    def forward(self, params=None):
        # Samples a tree, ensuring our stored parameters get substituted
        # into every node that is generated.
        if params is None:
            params = self.get_default_param_dict()

        scene_tree = SceneTree()
        root_node = self.root_node_type.init_with_default_parameters()
        self._instantiate_node_from_mean_field(root_node, params)
        scene_tree.add_node(root_node)
        return self._generate_from_node_recursive(scene_tree, root_node, params)

    def score(self, scene_tree, params=None, detach=False):
        ''' Scores given tree under this grammar using the grammar's currently
        stored parameter values, but the scene tree node's stored attributes.'''
        if params is None:
            params = self.get_default_param_dict()

        total_ll = 0.
        for node in scene_tree.nodes:
            # Score node attributes vs our mean fields.
            derived_var_dists, local_var_dists = self._make_variable_mean_fields(type(node), params)
            def calc_score(value_dict, dist_dict):
                # Detach the values
                if detach:
                    detached_value_dict = {key: value.detach() for key, value in value_dict.items()}
                    return node.get_variable_ll_given_dicts(detached_value_dict, dist_dict)
                else:
                    return node.get_variable_ll_given_dicts(value_dict, dist_dict)
            total_ll = total_ll + calc_score(
                node.get_derived_variable_values(),
                derived_var_dists
            )
            total_ll = total_ll + calc_score(
                node.get_local_variable_values(),
                local_var_dists
            )

            # Score node children.
            if isinstance(node, NonTerminalNode):
                children = list(scene_tree.successors(node))
                inclusion_dist = self._make_inclusion_dist(node, params)
                inclusion_value = node.get_child_indicator_vector([type(c) for c in children])
                total_ll = total_ll + inclusion_dist.log_prob(inclusion_value)

        return total_ll


class SceneTree(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)

    def find_node_by_name(self, name):
        for node in self.nodes:
            if node.name == name:
                return node
        raise ValueError("No node named %s" % name)

    def find_nodes_by_type(self, target_type):
        nodes = []
        for node in self.nodes:
            if isinstance(node, target_type):
                nodes.append(node)
        return nodes

    def get_node_parent_or_none(self, node):
        parents = list(self.predecessors(node))
        if len(parents) == 0:
            return None
        elif len(parents) == 1:
            return parents[0]
        else:
            raise ValueError("Bad scene tree: node has more than one parent.", node)

    def get_recursive_children_of_node(self, node):
        nodes = list(self.successors(node))
        out = [node]
        for node in nodes:
            out += self.get_recursive_children_of_node(node)
        return out

    def resample_instantiations(self, root_node, root_node_instantiation_dict=None):
        ''' Resample the continuous parameters of all nodes under the given
        node in the tree from their priors, but keep the tree structure the same.

        If the root node instantiation dict is *not* provided, the root node is assumed
        to be instantiated and will not be re-instantiated. '''
        raise NotImplementedError("Not trustworthy")
        assert root_node in self.nodes
        if root_node_instantiation_dict:
            root_node.instantiate(root_node_instantiation_dict)
        else:
            assert root_node.instantiated
        node_queue = [root_node]
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            if isinstance(node, NonTerminalNode):
                children = list(self.successors(node))
                node.instantiate_children(children)
                node_queue += children

    def resample_subtree(self, root_node, root_node_instantiation_dict=None):
        ''' Completely resamples the subtree rooted at the given root node. If the
        root node instantiating dict is supplied, resamples the root node local variables
        too; otherwise asserts that it's already instantiated. '''
        raise NotImplementedError("Not trustworthy")
        assert root_node in self.nodes

        # Immediately rename root node so that attempts to resample suceed.
        old_root_name = root_node.name
        root_node.name += "_resampled"
        
        if root_node_instantiation_dict:
            root_node.instantiate(root_node_instantiation_dict)
        else:
            assert root_node.instantiated
        
        new_subtree = SceneTree.forward_sample_from_root(root_node)
        # Remove the existing subtree...
        node_queue = list(self.successors(root_node))
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            node_queue += self.successors(node)
            self.remove_node(node)
        # and add the new one in its place.
        node_queue = [root_node]
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            children = new_subtree.successors(node)
            for child in children:
                self.add_node(child)
                self.add_edge(node, child)
            node_queue += children
        # Done!
        return root_node

    def get_subtree_log_prob(self, root_node, include_continuous=True, include_discrete=True):
        assert root_node in self.nodes
        node_queue = [root_node]
        ll = 0.
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            if include_continuous:
                ll += node.get_continuous_variable_ll()
            if isinstance(node, NonTerminalNode):
                children = self.successors(node)
                if include_discrete:
                    ll += node.get_children_ll()
                node_queue += children
        return ll

    def get_log_prob(self, **kwargs):
        return self.get_subtree_log_prob(
            get_tree_root(self), **kwargs)
