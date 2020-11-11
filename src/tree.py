import networkx as nx

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
    def generate_from_root_type(root_node_type):
        ''' Generates an unconditioned parse tree from a given root node type. '''
        assert issubclass(root_node, RootNode)
        # Start by sampling the root node itself.
        unexpanded_nodes = [ root_node.sample() ]
        parse_tree = ParseTree()
        parse_tree.add_node(root_node)
        while len(input_nodes_with_parents)>  0:
            parent_node = input_nodes_with_parents.pop(0)
            if isinstance(parent_node, TerminalNode):
                # Nothing more to do with this node
                pass
            else:
                new_nodes = parent_node.sample_products()
                for new_node in new_nodes:
                    parse_tree.add_node(new_node)
                    parse_tree.add_edge(parent_node, new_node)
                unexpanded_nodes += new_nodes
        return parse_tree