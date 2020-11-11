import pyro
import pyro.distributions as dist

class Node(object):
    ''' Most abstract form of a node / symbol in the grammar. '''
    def __init__(self, name):
        self.name = name


class NonTerminalNode(Node):
    ''' Abstract interface for nonterminal nodes, which are responsible
    for sampling a set of production rules to produce new nodes. '''

    def __init__(self, name, product_set):
        '''
        Args:
            name: String name for this terminal node.
            product_set: A list of Node classes that this node
                can generate.
        '''
        self.product_set = product_set
        Node.__init__(self, name)

    def _sample_discrete(self):
        '''
        Should be implemented by a subclass, expressing stochasticity
        with Pyro sample primitives.
        
        Returns: A list of indices into the product set of this
        node, indicating the set of nodes to be generated.
        '''
        raise NotImplementedError()

    def _sample_continuous(self, product_list):
        '''
        Should be implemented by a subclass, expressing stochasticity
        with Pyro sample primitives.

        Args:
            product_list: A list of indices (from _sample_discrete)
            of product nodes to be generated.
        Returns:
            A list of instantiated product (non-root) Nodes.
        '''
        raise NotImplementedError()

    def sample_products(self):
        ''' 
            Returns: list of instantiated non-root Nodes.
        '''
        product_choices = _sample_discrete()
        return  _sample_continuous(product_choices)


class TerminalNode(Node):
    ''' The leafs of a generated scene tree will be terminal nodes.'''
    pass


class RootNode(NonTerminalNode):
    ''' A given grammar will have a single root node type. This should
    be able to be generated on its own. '''
    @staticmethod
    def sample():
        ''' Generate an instance of this root node. '''
        raise NotImplementedError()


# TODO(gizatt) There's no way this crazy multiple inheritence is a good
# idea.
class OrNodeMixin(NonTerminalNode):
    ''' Provides an implementation of _sample_discrete that randomly
    chooses exactly one of the available products to generate. '''
    def __init__(self, production_weights):
        if len(production_weights) != len(self.product_set):
            raise ValueError("# of potential products and weights must match.")
        if len(production_weights) == 0:
            raise ValueError("Must have nonzero number of production rules.")
        self.production_weights = production_weights
        self.production_dist = dist.Categorical(production_weights)

    def _sample_discrete(self):
        if not hasattr(self, self.production_dist):
            raise ValueError("OrNodeMixin.__init__ was not called.")
        active_rule = pyro.sample(self.name + "_or_sample", self.production_dist)
        return [self.product_set[active_rule]]


class AndNodeMixin(NonTerminalNode):
    ''' Provides an implementation of _sample_discrete that deterministically
    activates all available product generations. '''
    def sample_production_rules(self, parent):
        return self.product_set

# class CovaryingSetNode(NonTerminalNode):
#     ''' Convenience specialization of a nonterminal node: given a set of
#     production rules, can choose any combination of those rules. '''
#     @staticmethod
#     def build_init_weights(num_production_rules, production_weights_hints = {},
#                            remaining_weight = 1.):
#         '''
#             Helper to generate a valid set of production weights for this node type.
    
#             Args:
#                 num_production_rules: The total number of production rules.
#                 production_weights_hints: A dict keyed by tuples of ints (representing
#                     the set of production rules, as indexes into the rule list), with
#                     float values.
#                 remaining_weight: The weight to assign to each unspecified rule combination.

#             Returns:
#                 A valid, normalized init_weights vector with 2^(num_production_rules)
#                 entries as a torch tensor.
#         '''
#         assert(remaining_weight >= 0.)
#         num_combinations = 2**num_production_rules
#         init_weights = torch.ones(num_combinations).double() * remaining_weight
#         for hint in production_weights_hints.keys():
#             val = production_weights_hints[hint]
#             assert(val >= 0.)
#             combination_index = 0
#             for index in hint:
#                 assert(isinstance(index, int) and index >= 0 and
#                        index < num_production_rules)
#                 combination_index += 2**index
#             init_weights[combination_index] = val
#         init_weights /= torch.sum(init_weights)
#         return init_weights
        
#     def __init__(self, name, production_rules, init_weights):
#         ''' Make a categorical distribution over
#            every possible combination of production rules
#            that could be active, with a separate weight
#            for each combination. (2^n weights!)'''
#         self.production_rules = production_rules
#         self.exhaustive_set_weights = init_weights
#         self.production_dist = dist.Categorical(
#             logits=torch.log(self.exhaustive_set_weights / (1. - self.exhaustive_set_weights)))
#         NonTerminalNode.__init__(self, name=name)

#     def sample_production_rules(self, parent):
#         # Select out the appropriate rules
#         selected_rules = pyro.sample(
#             self.name + "_exhaustive_set_sample",
#             self.production_dist)
#         output = []
#         for k, rule in enumerate(self.production_rules):
#             if (selected_rules >> k) & 1:
#                 output.append(rule)
#         return output


# class IndependentSetNode(NonTerminalNode):
#     ''' Convenience specialization of a nonterminal node: given a set of
#     production rules, can activate each rule as an independent Bernoulli
#     trial (with specified probabilities of activation). '''
#     def __init__(self, name, production_rules,
#                  production_probs):
#         if len(production_probs) != len(production_rules):
#             raise ValueError("Must have same number of production probs "
#                              "as rules.")
#         self.production_probs = production_probs
#         self.production_dist = dist.Bernoulli(production_probs).to_event(1)
#         self.production_rules = production_rules
#         NonTerminalNode.__init__(self, name=name)

#     def sample_production_rules(self, parent):
#         selected_rules = pyro.sample(
#             self.name + "_independent_set_sample",
#             self.production_dist)
#         # Select out the appropriate rules
#         output = []
#         for k, rule in enumerate(self.production_rules):
#             if selected_rules[k] == 1:
#                 output.append(rule)
#         return output


# class IndependentSetNode(NonTerminalNode):
#     ''' Convenience specialization of a nonterminal node: given a set of
#     production rules, can activate each rule as an independent Bernoulli
#     trial (with specified probabilities of activation). '''
#     def __init__(self, name, production_rules,
#                  production_probs):
#         if len(production_probs) != len(production_rules):
#             raise ValueError("Must have same number of production probs "
#                              "as rules.")
#         self.production_probs = production_probs
#         self.production_dist = dist.Bernoulli(production_probs).to_event(1)
#         self.production_rules = production_rules
#         NonTerminalNode.__init__(self, name=name)

#     def sample_production_rules(self, parent):
#         selected_rules = pyro.sample(
#             self.name + "_independent_set_sample",
#             self.production_dist)
#         # Select out the appropriate rules
#         output = []
#         for k, rule in enumerate(self.production_rules):
#             if selected_rules[k] == 1:
#                 output.append(rule)
#         return output
