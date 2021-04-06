from functools import partial
import networkx as nx

import pyro
from pyro.contrib.autoname import scope, name_count
import pyro.distributions as dist
import torch

from .nodes import NonTerminalNode, TerminalNode, Node, NodeParameter
from .rules import ProductionRule


def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node


class SceneGrammar(torch.nn.Module):
    '''
    Manages a grammar: given a root node type, provides
    ability to sample trees under given parameter settings.
    '''

    def __init__(self, root_node_type):
        super().__init__()
        self.root_node_type = root_node_type
        self.params_by_node_type = {}
        # But our database of what parameters exist for each node type.
        for node_type in self.get_all_types_in_grammar(root_node_type):
            params = node_type().get_parameters()
            self.params_by_node_type[node_type] = params
            for name, param in params.items():
                self.register_parameter("%s:%s" % (node_type.__name__, name), param.get_unconstrained_value())

    @staticmethod
    def get_all_types_in_grammar(root_node_type):
        all_types = set()
        input_queue = [root_node_type]
        while len(input_queue) > 0:
            curr_type = input_queue.pop(0)
            all_types.add(curr_type)
            if issubclass(curr_type, NonTerminalNode):
                for new_type in curr_type().get_maximal_child_type_list():
                    if new_type not in all_types:
                        input_queue.append(new_type)
        return all_types

    def _apply_param_override(self, node):
        # Set the specified node's params from the grammar's param store.
        # TODO(gizatt) This is pretty awful. There must be a refactor,
        # probably to the node API, to make this more natural. Something
        # about containing exactly where the params get allocated and who
        # they belong to. Maybe they should be passed in via the
        # constructor, and kept track of at the grammar level?
        assert type(node) in self.params_by_node_type.keys()
        for name, param in self.params_by_node_type[type(node)].items():
            assert hasattr(node, name) and isinstance(getattr(node, name), NodeParameter)
            print("Overriding %s (%s) -> %s" % (name, getattr(node, name).get_value(), param.get_value()))
            setattr(node, name, param)

    def _generate_from_node_recursive(self, scene_tree, parent_node):
        if isinstance(parent_node, TerminalNode):
            return scene_tree
        else:
            # Choose what gets generated.
            children = parent_node.sample_children()
            for child in children:
                self._apply_param_override(child)
            # Do the actual generation of local (continuous) variables.
            parent_node.instantiate_children(children)
            for child_node in children:
                scene_tree.add_node(child_node)
                scene_tree.add_edge(parent_node, child_node)
                if isinstance(child_node, NonTerminalNode):
                    scene_tree = self._generate_from_node_recursive(scene_tree, child_node)
        return scene_tree

    def forward(self, root_node_instantiation_dict):
        # Samples a tree, ensuring our stored parameters get substituted
        # into every node that is generated.
        scene_tree = SceneTree()
        root_node = self.root_node_type()
        self._apply_param_override(root_node)
        root_node.instantiate(root_node_instantiation_dict)
        scene_tree.add_node(root_node)
        return self._generate_from_node_recursive(scene_tree, root_node)

    def get_tree_generation_log_prob(self, scene_tree):
        ''' Scores given tree under this grammar using the currently stored
        parameter values and node attributes. '''

        # To ensure the local parameters take effect
        total_score = torch.tensor([0.])
        node_queue = [get_tree_root(scene_tree)]
        while len(node_queue) > 0:
            curr_node = node_queue.pop(0)


        return torch.tensor([0.])
        
    @staticmethod
    def make_meta_scene_tree(root_node_type):
        ''' Given a root node, generates a meta-tree of node types (without
        continuous variables) for which any generated tree from this root is
        a subgraph (again not considering continuous variables). '''
        meta_tree = nx.DiGraph()
        root_node = root_node_type()
        meta_tree.add_node(root_node)
        node_queue = [root_node]
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            new_node_types = node.get_maximal_child_type_list()
            for new_node_type in new_node_types:
                new_node = new_node_type()
                meta_tree.add_node(new_node)
                meta_tree.add_edge(node, new_node)
                if isinstance(new_node, NonTerminalNode):
                    node_queue.append(new_node)
        return meta_tree


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

    def get_trace(self):
        ''' Returns a pyro trace of the forward sampling of this tree.
        This is reconstructed from each individual node in the tree, so if you've
        modified / resampled subtrees, this will still rebuild a correct overall trace. '''
        # This is a little hard -- the "easy" way to do it would be
        # to collect all sample site values from all nodes, and then
        # re-run forward_sample_from_root. But that'd create brand new
        # nodes, and we'd rather preserve our existing nodes (so we keep
        # the same names). Instead, I'll build a routine that manually
        # calls every node's sample and instantiate methods, but throwing
        # out the new nodes and keeping the old ones.
        # Still super funky... Pyro and graph-structured data like this
        # just don't play very well, do they?
        raise NotImplementedError("Needs reimplementation -- doesn't handle instantiation of local attributes correctly.")
        def trace_to_observe_dict(trace):
            return {key: site["value"] for key, site in trace.nodes.items()
                    if site["type"] is "sample"}
        def resample_tree_in_place(scene_tree):
            # The root is the only node that doesn't get instantiated
            # by its parent, so call that manually.
            root = get_tree_root(self)
            assert root.instantiated
            pyro.poutine.condition(
                root.instantiate,
                data=trace_to_observe_dict(root.instantiate_trace)
            )(root.derived_attributes)
            # Then, for every node, "re-run" (with fixed output) the
            # child resampling and instantiation.
            for node in scene_tree.nodes():
                if isinstance(node, NonTerminalNode):
                    assert node.children_sampled
                    pyro.poutine.condition(node.sample_children, 
                        data=trace_to_observe_dict(node.sample_children_trace))()
                    children = list(scene_tree.successors(node))
                    pyro.poutine.condition(
                        node.instantiate_children,
                        data=trace_to_observe_dict(node.instantiate_children_trace)
                    )(children=children)
            return scene_tree
        return pyro.poutine.trace(resample_tree_in_place).get_trace(self)

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
