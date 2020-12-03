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
    def __init__(self):
        nx.DiGraph.__init__(self)

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
                        new_nodes = rule.sample_products(parent_node)
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
        root_node = root_node_type(**kwargs)
        parse_tree = SceneTree()
        parse_tree.add_node(root_node)
        return SceneTree._generate_from_node_recursive(parse_tree, root_node)

