from functools import partial
import networkx as nx

import pyro
from pyro.contrib.autoname import scope, name_count

from .nodes import NonTerminalNode, TerminalNode, Node
from .rules import ProductionRule


def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node


class SceneTree(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)

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
        assert root_node in self.nodes
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

    def get_subtree_log_prob(self, root_node, include_instantiate=True, include_topology=True):
        assert root_node in self.nodes
        node_queue = [root_node]
        ll = 0.
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            if include_instantiate:
                ll += node.get_instantiate_ll()
            if isinstance(node, NonTerminalNode):
                children = self.successors(node)
                if include_topology:
                    ll += node.get_children_ll()
                if include_instantiate:
                    ll += node.get_instantiate_children_ll()
                node_queue += children
        return ll

    def get_log_prob(self, **kwargs):
        return self.get_subtree_log_prob(
            get_tree_root(self), **kwargs)

    @staticmethod
    def _generate_from_node_recursive(parse_tree, parent_node):
        if isinstance(parent_node, TerminalNode):
            return parse_tree
        else:
            # Choose what gets generated.
            children = parent_node.sample_children()
            # Do the actual generation of local (continuous) variables.
            parent_node.instantiate_children(children)
            for child_node in children:
                parse_tree.add_node(child_node)
                parse_tree.add_edge(parent_node, child_node)
                if isinstance(child_node, NonTerminalNode):
                    parse_tree = SceneTree._generate_from_node_recursive(parse_tree, child_node)
        return parse_tree

    @staticmethod
    def forward_sample_from_root(root_node):
        '''
        Generates an unconditioned parse tree from an instantiated
        root node type.
        '''
        assert root_node.instantiated
        parse_tree = SceneTree()
        parse_tree.add_node(root_node)
        return SceneTree._generate_from_node_recursive(parse_tree, root_node)

    @staticmethod
    def forward_sample_from_root_type(root_node_type, root_node_instantiation_dict):
        root = root_node_type()
        root.instantiate(root_node_instantiation_dict)
        return SceneTree.forward_sample_from_root(root)

    @staticmethod
    def make_meta_scene_tree(root_node):
        ''' Given a root node, generates a meta-tree of node types (without
        continuous variables) for which any generated tree from this root is
        a subgraph (again not considering continuous variables). '''
        meta_tree = nx.DiGraph()
        meta_tree.add_node(root_node)
        node_queue = [root_node]
        while len(node_queue) > 0:
            node = node_queue.pop(0)
            new_nodes = node.get_maximal_child_list()
            for new_node in new_nodes:
                meta_tree.add_node(new_node)
                meta_tree.add_edge(node, new_node)
                if isinstance(new_node, NonTerminalNode):
                    node_queue.append(new_node)
        return meta_tree


    # PARSING AND PARSING UTILITIES
    @staticmethod
    def parse_greedily_from_partial_tree(root_node_type, partial_tree):
        ''' Given a partial tree (an nx digraph) in a state where the tree may not be feasible,
        but where the supplied nodes are fully realized: so we need to fix that
        non-root nodes may not have parents, and non-terminal nodes may not
        have appropriate children.

        To fix this, we:
        1) For all fully-realized non-terminal nodes in the tree, unconditionally
        forward-sample their subtree. Now the tree only needs to be fixed "upwards."
        2) Repeatedly randomly select subsets of orphan nodes (non-root parentless nodes)
        and randomly choose new parents for them from the set of current + possible nodes.

        '''
        raise NotImplementedError()

