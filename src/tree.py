import networkx as nx
from .nodes import RootNode, TerminalNode
from pyro.contrib.autoname import scope, name_count

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