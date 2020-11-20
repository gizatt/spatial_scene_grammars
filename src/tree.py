import networkx as nx
from .nodes import RootNode, TerminalNode, Node
from .rules import ProductionRule
from pyro.contrib.autoname import scope, name_count

def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node

class ParseTree(nx.DiGraph):
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
        for node in nodes:
            nodes += self.get_recursive_children_of_node(node)
        return nodes

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
    @name_count
    def generate_from_root_type(root_node_type):
        ''' Generates an unconditioned parse tree from a root node class. '''
        assert issubclass(root_node_type, RootNode)
        unexpanded_nodes = [ root_node_type.sample() ]
        parse_tree = ParseTree()
        parse_tree.add_node(unexpanded_nodes[0])
        while len(unexpanded_nodes)>  0:
            parent_node = unexpanded_nodes.pop(0)
            if isinstance(parent_node, TerminalNode):
                # Nothing more to do with this node
                pass
            else:
                # Expand by picking a production rule
                production_rules = parent_node.sample_production_rules()
                for i, rule in enumerate(production_rules):
                    parse_tree.add_node(rule)
                    parse_tree.add_edge(parent_node, rule)
                    new_nodes = rule.sample_products(parent_node)
                    for new_node in new_nodes:
                        parse_tree.add_node(new_node)
                        parse_tree.add_edge(rule, new_node)
                    unexpanded_nodes += new_nodes
        return parse_tree