import networkx as nx
from .nodes import TerminalNode, Node
from .rules import ProductionRule
from pyro.contrib.autoname import scope, name_count

def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node

class SceneTree(nx.DiGraph):
    def __init__(self, root_node_type):
        nx.DiGraph.__init__(self)
        self._used_node_names = set()
        self._root_node_type = root_node_type

    # ACCESSORS AND BASICS
    def get_node_parent_or_none(self, node):
        parents = list(self.predecessors(node))
        if len(parents) == 0:
            return None
        elif len(parents) == 1:
            return parents[0]
        else:
            print("Bad parse tree: ", self)
            print("Node: ", node)
            print("Parents: ", parents)
            raise NotImplementedError("> 1 parent --> bad parse tree")

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

    def get_tree_without_production_rules(self):
        # Returns a raw nx.DiGraph of the same nodes
        # in this tree, but with the production rules removed
        # (so nodes are connected directly to their parent node).
        new_tree = nx.DiGraph()
        for node in self.nodes:
            if isinstance(node, Node):
                new_tree.add_node(node)
        for node in self.nodes:
            if isinstance(node, ProductionRule):
                parent = self.get_node_parent_or_none(node)
                assert(parent is not None)
                for child in list(self.successors(node)):
                    new_tree.add_edge(parent, child)
        return new_tree

    # GENERATION AND GENERATION UTILITIES
    def get_unique_name_for_node_type(self, node_type):
        # Generates a unique name for a new instance of a given
        # node type.
        k = 0
        def get_candidate_name(l):
            return "%s_%d" % (node_type.__name__, l)
        name = get_candidate_name(k)
        while name in self._used_node_names:
            k += 1
            name = get_candidate_name(k)
        self._used_node_names.add(name)
        return name

    @staticmethod
    def _generate_from_node_recursive(parse_tree, parent_node):
        if isinstance(parent_node, TerminalNode):
            return parse_tree
        else:
            with scope(prefix=parent_node.name):
                with scope(prefix="rules"):
                    production_rules = parent_node.sample_production_rules()
                for i, rule in enumerate(production_rules):
                    parse_tree.add_node(rule)
                    parse_tree.add_edge(parent_node, rule)

                    with scope(prefix="prod_%d" % i):
                        new_node_names = [parse_tree.get_unique_name_for_node_type(node_type)
                                          for node_type in rule.child_types]
                        new_nodes = rule.sample_products(parent_node, new_node_names)
                        for new_node in new_nodes:
                            parse_tree.add_node(new_node)
                            parse_tree.add_edge(rule, new_node)
                            parse_tree = SceneTree._generate_from_node_recursive(parse_tree, new_node)

        return parse_tree

    @staticmethod
    def forward_sample_from_root_type(root_node_type, **kwargs):
        '''
        Generates an unconditioned parse tree from a root node type
        and a list of any arguments required to instantiate it.
        '''
        parse_tree = SceneTree(root_node_type=root_node_type)
        if "name" not in kwargs.keys():
            kwargs["name"] = parse_tree.get_unique_name_for_node_type(root_node_type)
        else:
            parse_tree._used_node_names.add(kwargs["name"])
        root_node = root_node_type(**kwargs)
        parse_tree.add_node(root_node)
        return SceneTree._generate_from_node_recursive(parse_tree, root_node)

    @staticmethod
    def make_meta_scene_tree(root_node_type):
        ''' Given a root node, generates a meta-tree of node types (without
        continuous variables) for which any generated tree from this root is
        a subgraph (again not considering continuous variables). '''
        meta_tree = nx.DiGraph()
        input_types = [root_node_type]
        while len(input_types) > 0:
            cur_type = input_types.pop(0)
            new_types = cur_type.get_maximal_child_type_list()
            for new_type in new_types:
                meta_tree.add_edge(cur_type, new_type)
                if isinstance(new_type, NonTerminalNode):
                    input_types.append(new_type)
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