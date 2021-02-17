import networkx as nx
from .nodes import NonTerminalNode, TerminalNode, Node
from .rules import ProductionRule
from pyro.contrib.autoname import scope, name_count


class SceneTree(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)

    def find_nodes_by_type(self, target_type):
        nodes = []
        for node in self.nodes:
            if isinstance(node, target_type):
                nodes.append(node)
        return nodes

    def get_recursive_children_of_node(self, node):
        nodes = list(self.successors(node))
        out = [node]
        for node in nodes:
            out += self.get_recursive_children_of_node(node)
        return out

    @staticmethod
    def _generate_from_node_recursive(parse_tree, parent_node):
        if isinstance(parent_node, TerminalNode):
            return parse_tree
        else:
            # Choose what gets generated.
            children = parent_node.sample_children()
            # Do the actual generation of local (continuous) variables.
            parent_node.instantiate_children(children)
            for child in children:
                if isinstance(child, NonTerminalNode):
                    parse_tree = SceneTree._generate_from_node_recursive(parse_tree, child)
        return parse_tree

    @staticmethod
    def forward_sample_from_root_type(root_node):
        '''
        Generates an unconditioned parse tree from an instantiated
        root node type.
        '''
        assert root_node.instantiated
        parse_tree = SceneTree()
        parse_tree.add_node(root_node)
        return SceneTree._generate_from_node_recursive(parse_tree, root_node)

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

