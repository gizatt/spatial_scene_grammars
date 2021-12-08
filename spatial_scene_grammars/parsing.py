'''
Scene Parsing utilities for this grammar

Given the grammar description (the set of node types and their
rules) and an observed set of nodes in the language, can we
recover likely parse trees?

We provide a few approaches:

1) MIP MAP Parsing: Parameterize the space of parse trees by
forming a "super tree" for the grammar. Create binary activation
variables and continuous pose variables for each node that the grammar
could possibly produce. Optimize those variables to form a
maximum-probability tree that explains the observed nodes.

Depending on the types of rules in the grammar, this approach returns
highly optimized parse trees pretty efficiently, but it becomes inefficient
when there are many nodes with unknown rotations. It does *not* require
proposal generation for unobserved intermediate nodes.


2) MIP MAP Parsing with greedy proposals: Instead of optimizing
over the poses of unobserved nodes that could participate in the parse
tree, we instead use a set of heuristic methods to propose reasonable
intermediate nodes, and write a binary-only MIP that decides how to
link together the grammar root, the observed nodes, and a subset of
those intermediate nodes to create a feasible, maximal-score parse tree.

We provide some additional utilities:

1) Nonlinear-optimized based refinement of a parse tree: given a parse
tree, optimize its node poses while keeping the same tree structure.

'''

from datetime import datetime
import time
import networkx as nx
import numpy as np
import sys
from copy import deepcopy
from collections import namedtuple
from functools import partial
import logging

import pydrake
from pydrake.all import (
    AngleAxis,
    AngleAxis_,
    ClpSolver,
    CommonSolverOption,
    Expression,
    MathematicalProgram,
    MakeSolver,
    MixedIntegerBranchAndBound,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    RotationMatrix_,
    GurobiSolver,
    SnoptSolver,
    OsqpSolver,
    Solve,
    SolverOptions,
    SolutionResult,
    VPolytope,
    MixedIntegerRotationConstraintGenerator,
    IntervalBinning,
    UniformlyRandomRotationMatrix,
    Variable
)

from .nodes import *
from .rules import *
from .scene_grammar import *
from .drake_interop import *

'''
###############################################################################
###                         Common helper functions.
###
###############################################################################
'''
def prepare_grammar_for_parsing(prog, grammar, optimize_parameters=False, inequality_eps=1E-6):
    # Populates grammar.rule_params_by_node_type_optim, which has parallel
    # structure to grammar.rule_params_by_node_type, which with appropriately
    # constrained decision variables if optimize_parameters is True, or
    # placeholder numpy values of the current parameter setting otherwise.

    def _make_constrained_param(param, name):
        assert isinstance(param, ConstrainedParameter)
        val = param().detach().cpu().numpy()
        if optimize_parameters:
            new_vars = prog.NewContinuousVariables(*val.shape, name)
            if isinstance(param.constraint, constraints._Real):
                # No constraint necessary
                pass
            elif isinstance(param.constraint, constraints._Interval):
                prog.AddBoundingBoxConstraint(
                    param.constraint.lower_bound.detach().cpu().numpy(),
                    param.constraint.upper_bound.detach().cpu().numpy(),
                    new_vars
                )
            elif isinstance(param.constraint, constraints._Positive):
                for var in new_vars:
                    prog.AddLinearConstraint(prog >= inequality_eps)
            else:
                raise NotImplementedError("Constraint type %s" % param.constraint)
            return new_vars
        else:
            return val

    grammar.rule_params_by_node_type_optim = {}
    for node_type in grammar.all_types:
        param_dicts_by_rule = grammar.rule_params_by_node_type[node_type.__name__]
        grammar.rule_params_by_node_type_optim[node_type.__name__] = []
        rules = node_type.generate_rules()
        for k, (rule, (xyz_params, rot_params)) in enumerate(zip(rules, param_dicts_by_rule)):
            prefix = "%s:%s(%d):" % (node_type.__name__, type(rule).__name__, k)
            xyz_var_dict = {}
            for key, param in xyz_params.items():
                xyz_var_dict[key] = _make_constrained_param(param, name=prefix+key)
            rot_var_dict = {}
            for key, param in rot_params.items():
                rot_var_dict[key] = _make_constrained_param(param, name=prefix+key)
            grammar.rule_params_by_node_type_optim[node_type.__name__].append(
                [xyz_var_dict, rot_var_dict]
            )
    return grammar


def make_equivalent_sets(super_tree):
    '''
    Figure out which connected sets of nodes in the supertree have poses
    that are  constrained to be exactly equal.
    '''
    equivalence_graph = nx.Graph()
    equivalence_graph.add_nodes_from(super_tree.nodes)
    for parent_node in super_tree.nodes:
        children = super_tree.get_children(parent_node)
        ## Get child rule list. Can't use get_children_and_rules
        # here since we're operating on a supertree, so the standard
        # scene tree logic for getting rules isn't correct.
        if isinstance(parent_node, RepeatingSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))
        for child_node, rule in zip(children, rules):
            # TODO(gizatt) We may have other rule types that, depending on
            # parameters, imply equality constraints. They could be included
            # here generally if we query the rule whether it is fully constraining
            # at the current parameter setting.
            if isinstance(rule.xyz_rule, SamePositionRule) and isinstance(rule.rotation_rule, SameRotationRule):
                equivalence_graph.add_edge(parent_node, child_node)
    equivalent_sets = [EquivalentSet(super_tree, nodes) for nodes in nx.connected_components(equivalence_graph)]
    return equivalent_sets

def make_equivalent_sets_for_R_and_t(super_tree):
    '''
    Figure out which connected sets of nodes in the supertree have translations
    [or rotations] that are constrained to be exactly equal.
    '''
    t_equivalence_graph = nx.Graph()
    R_equivalence_graph = nx.Graph()
    t_equivalence_graph.add_nodes_from(super_tree.nodes)
    R_equivalence_graph.add_nodes_from(super_tree.nodes)
    for parent_node in super_tree.nodes:
        children = super_tree.get_children(parent_node)
        ## Get child rule list. Can't use get_children_and_rules
        # here since we're operating on a supertree, so the standard
        # scene tree logic for getting rules isn't correct.
        if isinstance(parent_node, RepeatingSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))
        for child_node, rule in zip(children, rules):
            # TODO(gizatt) We may have other rule types that, depending on
            # parameters, imply equality constraints. They could be included
            # here generally if we query the rule whether it is fully constraining
            # at the current parameter setting.
            if isinstance(rule.xyz_rule, SamePositionRule):
                t_equivalence_graph.add_edge(parent_node, child_node)
            if isinstance(rule.rotation_rule, SameRotationRule):
                R_equivalence_graph.add_edge(parent_node, child_node)
    t_equivalent_sets = [EquivalentSet(super_tree, nodes) for nodes in nx.connected_components(t_equivalence_graph)]
    R_equivalent_sets = [EquivalentSet(super_tree, nodes) for nodes in nx.connected_components(R_equivalence_graph)]
    return t_equivalent_sets, R_equivalent_sets

class EquivalentSet():
    def __init__(self, super_tree, nodes):
        self.super_tree = super_tree
        self.nodes = list(nodes)
        self.always_observable = None

    def __iter__(self):
        return iter(self.nodes)

    def is_always_observable(self):
        # Minimal caching to not do slightly-expensive recomputation
        if self.always_observable is None:
            self.always_observable = self._equivalent_set_activity_implies_observability()
        return self.always_observable

    def _equivalent_set_activity_implies_observability(self):
        ''' For a list of nodes `equivalent_set` from the supertree for
        a grammar, return True if any node in the equivalent set being
        active implies that one of the observable nodes in the equivalent set
        is active; and False otherwise.

        Strategy: construct an LP, where the activation of each node is a variable
        constrained in [0, 1]. Add constraints between parent/child sets from the supertree
        constraining the activations to be reasonable under the node types. Constrain
        at least one non-observable node to be on, but all observable nodes to be off.
        If there is a solution to this, then activity doesn't imply observability,
        so return False. Otherwise, return True.

        TODO: Is LP sufficient, or do we need binary variables?
        '''

        num_observable_nodes = len([node for node in self.nodes if node.observed])
        if num_observable_nodes == 0:
            # Can never be observable, as there are no observed nodes.
            return False
        if num_observable_nodes == len(self.nodes):
            # Trivially observable
            return True

        prog = MathematicalProgram()
        # Create activations per node
        node_to_activation = {
            node: prog.NewBinaryVariables(1)[0]
            for node in self.nodes
        }
        # At least one observed node, but all other nodes off.
        nonobserved_sum = 0.
        for node, var in node_to_activation.items():
            if not node.observed:
                prog.AddBoundingBoxConstraint(0., 1., var)
                nonobserved_sum += var
            else:
                # Observes off
                prog.AddBoundingBoxConstraint(0., 0., var)
        if not isinstance(nonobserved_sum, float):
            prog.AddLinearConstraint(nonobserved_sum >= 1.)

        # Parent-child type relationships
        for parent_node, parent_var in node_to_activation.items():
            children = self.super_tree.get_children(parent_node)
            ## Get child rule list. Can't use get_children_and_rules
            # here since we're operating on a supertree, so the standard
            # scene tree logic for getting rules isn't correct.
            if isinstance(parent_node, AndNode):
                for child in children:
                    if child in node_to_activation.keys():
                        prog.AddLinearConstraint(node_to_activation[child] == parent_var)
            elif isinstance(parent_node, OrNode):
                child_var_sum = 0.
                child_set_is_complete = True
                for child in children:
                    if child in node_to_activation.keys():
                        child_var_sum += node_to_activation[child]
                    else:
                        child_set_is_complete = False
                if child_set_is_complete and not isinstance(child_var_sum, float):
                    prog.AddLinearConstraint(child_var_sum == parent_var)
                else:
                    # Don't know what other out-of-set children are doing, so
                    # deactivate them.
                    prog.AddLinearConstraint(child_var_sum <= parent_var)
            elif isinstance(parent_node, IndependentSetNode):
                for child in children:
                    if child in node_to_activation.keys():
                        prog.AddLinearConstraint(node_to_activation[child] <= parent_var)

            elif isinstance(parent_node, RepeatingSetNode):
                for child in children:
                    if child in node_to_activation.keys():
                        prog.AddLinearConstraint(node_to_activation[child] <= parent_var)
            elif isinstance(parent_node, TerminalNode):
                assert len(children) == 0
            else:
                raise ValueError("Unexpected node type: ", type(parent_node))

        # Try to solve
        result = GurobiSolver().Solve(prog)
        if result.is_success():
            return False
        else:
            return True

'''
###############################################################################
                     Heuristic proposal-based parsing.

   Given an observed node set and grammar, build a library of candidate
   intermediate nodes (i.e. unobserved nodes that might be necessary to
   explain observations using the grammar) by a combination of top-down
   sampling and bottom-up proposal generation. These intermediate nodes
   can be assembled together with the grammar root and observed nodes
   to form feasible trees by a (hopefully!) efficient MIP.

###############################################################################
'''
def get_score_of_orphan_set_under_parent(parent_node, existing_children, orphan_set):
    '''
    Given a parent node, its current children, and an additional set of nodes,
    try to add the orphan set to this parent's child set.
    Returns:
    - The log-prob score of the parent under this whole orphan set.
    - A dictionary mapping from orphan set nodes to their rule_k mapping
      into the parent ordered rule list, if the log-prob score isn't -inf.
    '''
    # Try to add the orphan set under the parent, returning
    # a log-prob score (or negative inf if infeasible).
    # Trivial short-circuit: terminal nodes can't take children.
    if isinstance(parent_node, TerminalNode):
        return torch.tensor(-np.inf), None

    full_child_set = existing_children + orphan_set

    # For each node in the orphan set, we need to resolve its mapping
    # into the parent rule list. Start by figuring out which rules are
    # occupied, and greedily map each child onto the remaining rules
    # if the types match.
    if isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
        all_rules = parent_node.rules
    elif isinstance(parent_node, (RepeatingSetNode,)):
        all_rules = [parent_node.rules[0] for k in range(parent_node.max_children)]
    else:
        raise NotImplementedError(type(parent_node))
    taken = np.zeros(len(all_rules), dtype=bool)
    for child_node in existing_children:
        assert child_node.rule_k is not None
        assert taken[child_node.rule_k] == False, [child.rule_k for child in existing_children]
        taken[child_node.rule_k] = True
    for child_node in orphan_set:
        assert child_node.rule_k is None
        for i, rule in enumerate(all_rules):
            if taken[i] == False and isinstance(child_node, rule.child_type):
                # Match!
                taken[i] = True
                child_node.rule_k = i
                break
        if child_node.rule_k is None:
            # We couldn't match this child, so return clean up
            # our modifications to the orphan set and return infeasible.
            for orphan_node in orphan_set:
                orphan_node.rule_k = None
            return torch.tensor(-np.inf), None
    # Now all children have rule assignments, so we can use the node
    # to score the child set, and use each rule to score the parent/child
    # pair.
    total_score = parent_node.score_child_set(full_child_set)
    for child_node in full_child_set:
        total_score = total_score + all_rules[child_node.rule_k].score_child(parent_node, child_node)
        
    orphan_rule_mappings = {orphan_node: orphan_node.rule_k for orphan_node in orphan_set}
    # Clean up our modifications to the orphan set.
    for orphan_node in orphan_set:
        orphan_node.rule_k = None
    return total_score, orphan_rule_mappings

def sample_randomly_sized_random_orphan_set(orphans):
    ''' Naive and simple way of selecting orphans to parent: randomly
    pick the # of orphans, and then randomly select that many from the
    orphan set (without replacement).
    This will tend to select smaller groups, and is very unlikely to sample
    all members of a large cluster. '''
    number_of_sampled_nodes = min(np.random.geometric(p=0.8), len(orphans))
    # Pick the orphan node set. 
    return np.random.permutation(orphans)[:number_of_sampled_nodes].tolist()

def attempt_tree_repair_in_place(tree_guess, root_node, candidate_intermediate_nodes,
                                 max_iterations):
    '''
         Given a partial parse tree and its root node
         and a list of candidate intermediate nodes, try
         to rebuild the tree by connecting parent-less non-root
         nodes to other nodes in the tree or new intermediate nodes.
    '''
    
    num_iterations = 0
    while (1):
        ## Collect orphan nodes.
        orphans = [node for node in tree_guess.nodes
                   if (node is not root_node and tree_guess.get_parent(node) is None)]

        if len(orphans) == 0:
            # We should have a complete, feasible tree!
            break
        if num_iterations >= max_iterations:
            logging.error("Exceeding iteration limit on greedy parsing attempt.")
            break

        ## Sample an orphan set randomly.
        # Pick the size of the orphan set randomly.
        orphan_set = sample_randomly_sized_random_orphan_set(orphans)
        
        ## Enumerate ways of adding this orphan set:
        #  - Parent this to an existing node.
        #  - Parent this to a new node.
        # And calculate a score for each one.
        PotentialConnection = namedtuple(
            "PotentialConnection",
            ["score", "parent_node", "orphan_rule_mapping"]
        )
        potential_connections = []
        for parent_node in tree_guess.nodes:
            if isinstance(parent_node, TerminalNode) or parent_node in orphan_set:
                # Skip some trivial cases.
                continue
            score, orphan_rule_mapping = get_score_of_orphan_set_under_parent(
                parent_node, tree_guess.get_children(parent_node), orphan_set
            )
            if torch.isfinite(score):
                potential_connections.append(
                    PotentialConnection(
                        score=score, parent_node=parent_node,
                        orphan_rule_mapping=orphan_rule_mapping
                    )
                )
        for parent_node in candidate_intermediate_nodes:
            score, orphan_rule_mapping = get_score_of_orphan_set_under_parent(
                parent_node, [], orphan_set
            )
            if torch.isfinite(score):
                potential_connections.append(
                    PotentialConnection(
                        score=score, parent_node=parent_node,
                        orphan_rule_mapping=orphan_rule_mapping
                    )

                )

        ## Pick from among the options.
        if len(potential_connections) > 0:
            if len(potential_connections) > 1:
                scores = torch.stack([conn.score for conn in potential_connections])
                # Rescale these values to the biggest one is 1, and use them as weights
                # in a Categorical draw.
                weights = torch.exp(scores - scores.max()).flatten()
                ind = dist.Categorical(weights).sample()
                new_connection = potential_connections[ind]
            else:
                new_connection = potential_connections[0]

            if new_connection.parent_node in candidate_intermediate_nodes:
                candidate_intermediate_nodes.remove(new_connection.parent_node)

            ## Add the resulting connection to the tree guess.
            for orphan_node, rule_k in new_connection.orphan_rule_mapping.items():
                orphan_node.rule_k = rule_k
                # Adding this edge will add the parent node into the
                # tree if it's not already there.
                tree_guess.add_edge(new_connection.parent_node, orphan_node)
                print("Connecting %s to %s" % (new_connection.parent_node, orphan_node))

        ## Get ready for another loop.
        num_iterations += 1
    return tree_guess

def generate_top_down_intermediate_nodes_by_supertree(grammar, nodes_to_expand, max_recursion_depth=10):
    '''
    Given a grammar and a set of nodes to expand, generates an additional set of
    unobserved nodes by forward-sampling N supertrees rooted at each candidate node and
    the grammar root. These supertrees terminate when they reach another
    observed node or reach a maximum tree depth.
    '''

    root_types = [type(n) for n in nodes_to_expand]
    root_tfs = [n.tf.detach() for n in nodes_to_expand]
    
    candidate_intermediate_nodes = []
    for root_type, root_tf in zip(root_types, root_tfs):
        super_tree = grammar.make_super_tree_from_root_node_type(
            root_type, root_tf, max_recursion_depth=max_recursion_depth,
            detach=True, terminate_at_observed=True
        )
        # Grab all of the non-root nodes in this tree.
        super_tree_root = super_tree.get_root()
        candidate_intermediate_nodes += [
            node for node in super_tree.nodes
            if node is not super_tree_root
        ]
    assert all([n.observed is False for n in candidate_intermediate_nodes])
    return candidate_intermediate_nodes

def generate_bottom_up_intermediate_nodes_by_inverting_rules(grammar, observed_nodes, max_recursion_depth=10):
    '''
    Given a grammar and a set of unobserve nodes, produces candidate unobserved
    nodes by proposing parents for the observed nodes (and their proposed parents,
    etc) by inverting those rule types that are obvious how to invert.
    Annotates with a "_produced_parent" field if it produces parents.
    '''

    def invert_rot_rule(child, rot_rule):
        # Return a rotation tensor or None.
        if type(rot_rule) is SameRotationRule:
            return torch.matmul(child.rotation, rot_rule.offset.T)
        elif type(rot_rule) is ParentFrameBinghamRotationRule:
            # Return child times (inverse rotation of) mode of offset distribution.
            R = quaternion_to_matrix(rot_rule.M[:, -1])
            return torch.matmul(child.rotation, R.T)
        elif type(rot_rule) is UniformBoundedRevoluteJointRule:
            angle_axis = rot_rule.axis * rot_rule.parameters["center"]
            R_offset = axis_angle_to_matrix(angle_axis.unsqueeze(0))[0, ...]
            return torch.matmul(child.rotation, R_offset.T)
        elif type(rot_rule) in (UnconstrainedRotationRule, WorldFrameBinghamRotationRule):
            # We have no dependence on the parent rotation here, so take a wild guess.
            return torch.eye(3)
        else:
            logging.warning("Not inverting rotation rule of type %s" % type(rot_rule))
            return None
    
    def invert_xyz_rule(child, xyz_rule, parent_rotation):
        # Return a translation tensor or None.
        if type(xyz_rule) is SamePositionRule:
            return child.translation - torch.matmul(parent_rotation, xyz_rule.offset)
        elif type(xyz_rule) is ParentFrameGaussianOffsetRule:
            mean_offset = xyz_rule.parameters["mean"]
            return child.translation - torch.matmul(parent_rotation, mean_offset)
        elif type(xyz_rule) is WorldFrameBBoxOffsetRule:
            return child.translation - xyz_rule.parameters["center"]
        elif type(xyz_rule) in (WorldFrameBBoxRule, WorldFrameGaussianOffsetRule):
            # We have no dependence on the parent translation here, so take a wild guess.
            return torch.zeros(3)
        else:
            logging.warning("Not inverting translation rule of type %s" % type(xyz_rule))
            return None

    def get_potential_parents_for_node(node):
        new_parents = []
        for node_type in grammar.all_types:
            prototype = node_type(torch.eye(4))
            if prototype.observed:
                continue
            for rule in prototype.rules:
                if isinstance(node, rule.child_type):
                    # This rule from this unobserved intermediate node could
                    # produce this observed node. For invertible rule types,
                    # create a candidate intermediate node at the right pose
                    # to have high score for this child.
                    parent_rotation = invert_rot_rule(node, rule.rotation_rule)
                    if parent_rotation is None:
                        continue
                    parent_translation = invert_xyz_rule(node, rule.xyz_rule, parent_rotation)
                    if parent_translation is None:
                        continue
                    prototype.rotation = parent_rotation
                    prototype.translation = parent_translation
                    new_parents.append(prototype)
        return new_parents
        
    expand_queue = [] + observed_nodes
    for node in expand_queue:
        node.__recursion_count = 0
    candidate_intermediate_nodes = []
    while len(expand_queue) > 0:
        node = expand_queue.pop(0)
        if node.__recursion_count > max_recursion_depth:
            continue
        new_nodes = get_potential_parents_for_node(node)
        if len(new_nodes) > 0:
            node._produced_parent = True
        for new_node in new_nodes:
            new_node.__recursion_count = node.__recursion_count + 1
            new_node._produced_parent = False
        expand_queue += new_nodes
        candidate_intermediate_nodes += new_nodes

    return list(set(candidate_intermediate_nodes))

def generate_intermediate_nodes_from_equivalent_sets(grammar, observed_nodes, max_recursion_depth=10):
    super_tree = grammar.make_super_tree(max_recursion_depth=max_recursion_depth, detach=True)

    # For translations and rotations separately, figure out which connected sets of nodes in the
    # supertree have translations [or rotations] that are constrained to be exactly equal.
    t_equivalent_sets, R_equivalent_sets = make_equivalent_sets(super_tree)

    # All nodes will be annotated with _t_possibilities and _R_possibilities based
    # on equivalent set-to-observation correspondences.
    for t_equivalent_set in t_equivalent_sets:
        t_observed_node_types = set([type(node) for node in t_equivalent_set if node.observed])
        set_t_possibilities = []
        for observed_node in observed_nodes:
            if type(observed_node) in t_observed_node_types:
                set_t_possibilities.append(observed_node.translation)
        for node in t_equivalent_set:
            node._t_possibilities = set_t_possibilities

    for R_equivalent_set in R_equivalent_sets:
        R_observed_node_types = set([type(node) for node in R_equivalent_set if node.observed])
        set_R_possibilities = []
        for observed_node in observed_nodes:
            if type(observed_node) in R_observed_node_types:
                set_R_possibilities.append(observed_node.rotation)
        for node in R_equivalent_set:
            node._R_possibilities = set_R_possibilities

    # For each unobserved node type in the supertree, produce proposals
    # at the equivalent set locations.
    candidate_intermediate_nodes = []
    for node in super_tree:
        if node.observed:
            continue
        # For unobserved nodes...
        for t in node._t_possibilities:
            for R in node._R_possibilities:
                new_node = deepcopy(node)
                new_node.translation = t
                new_node.rotation = R
                candidate_intermediate_nodes.append(new_node)

    return candidate_intermediate_nodes

def prune_node_set(node_set):
    '''
    Given a set of nodes, returns a de-duplicated set of nodes that are
    unique in terms of their combined type and pose.
    '''
    pruned_node_set = []
    for node in node_set:
        is_in_pruned_set = False
        for existing_node in pruned_node_set:
            if type(node) is type(existing_node) and torch.allclose(node.tf, existing_node.tf):
                is_in_pruned_set = True
                break
        if not is_in_pruned_set:
            pruned_node_set.append(node)

    return pruned_node_set

def generate_candidate_node_pose_sets(grammar, observed_nodes, max_recursion_depth=10, verbose=False):
    '''
    Given a grammar and a set of observed nodes (which are assumed to all be
    generate-able by the grammar), generates a set of candidate poses for each
    unobserved node type in the grammar by a bottom-up-then-top-down sampling procedure.

    Use a combination of approaches:
    1) Randomly sample top-down from each observed node + the scene root until hitting
    another observed node.
    2) Starting from observed nodes, propose new intermediate nodes bottom-up by
    inverting obvious-to-invert nodes.

    I've played with seeding top-down sampling from bottom-up proposed nodes and
    equivalent-set-based proposals (that tactic is now automatic in the parser
    machinery) but it quickly gets out of hand in terms of number of proposals for
    complex grammars. This set of rules is hopefully sufficient for many useful
    examples.
    '''

    # Randomly sample top-down from observed + root.
    all_known_fixed_nodes = prune_node_set(observed_nodes + [grammar.root_node_type(grammar.root_node_tf),])
    top_down_candidate_intermediate_nodes = generate_top_down_intermediate_nodes_by_supertree(
        grammar, all_known_fixed_nodes, max_recursion_depth=max_recursion_depth
    )
    if verbose:
        print("%d top-down candidates." % len(top_down_candidate_intermediate_nodes))

    # Generate bottom-up candidates.
    bottom_up_candidate_intermediate_nodes = generate_bottom_up_intermediate_nodes_by_inverting_rules(
        grammar, observed_nodes, max_recursion_depth=max_recursion_depth
    )
    bottom_up_candidate_intermediate_nodes = prune_node_set(bottom_up_candidate_intermediate_nodes)
    if verbose:
        print("%d bottom-up candidates." % len(bottom_up_candidate_intermediate_nodes))

    # Prune again to get a complete set of unique proposed nodes.
    all_candidate_nodes = prune_node_set(
        top_down_candidate_intermediate_nodes +
        bottom_up_candidate_intermediate_nodes
    )
    print("Post final pruning: ", len(all_candidate_nodes))
    
    # Convert to a mapping from node type to node pose for non-observed nodes.
    proposed_poses_by_type = {}
    for node in all_candidate_nodes:
        if node.observed or type(node) is grammar.root_node_type:
            continue
        type_name = type(node).__name__
        if type_name not in proposed_poses_by_type.keys():
            proposed_poses_by_type[type_name] = []
        proposed_poses_by_type[type_name].append(node.tf)

    return proposed_poses_by_type

def sample_likely_tree_with_greedy_parsing(
        grammar, observed_nodes, max_recursion_depth=10,
        max_attempts=10, max_iterations_per_attempt=100, verbose=False):
    ''' Given a grammar and an observed node set, find a posterior-likely tree
    from the grammar that explains those observed nodes by growing a parse tree
    bottom-up from the observations. Possible ways of explaining a given node
    are found by searching over nodes in the current tree and a population
    of proposed intermediate nodes as potential parents.


    The algorithm is reimplemented (and hopefully improved) from the one
    used in [Izatt & Tedrake '20]:
    1) Initialize a partial scene tree that contains only grammar root node + pose,
    the set of observed nodes, each with no connections. Sample a few supertrees
    from the grammar and collect a population of sampled intermediate node
    poses for each unobserved node pose.
    2) Collect the current set of orphan nodes that need to be assigned parents.
    3) Randomly sample a set of orphan nodes to attempt to parent.
       OPPORTUNITY FOR EXTENSION: Sample these "intelligently" by forming orphan
       node affinities based on their relative poses, and the typical relative poses
       of objects of that type from a population of sampled scenes.
    4) For that orphan set, enumerate all ways that they could be explained by
       iterating over nodes in the candidate tree + proposal set, and checking if
       the orphan set can be appended to the existing child set of the node. If
       the parent node is already instantiated, then apply a score based on its
       current pose.
       OPPORTUNITY FOR EXTENSION: Use optimization to improve the parent (and maybe
       child) poses here.
    5) Sample the parent connection to commit, using the scores as sampling weights,
       and make the appropriate modification to the tree.
    6) Loop to 2 until there are no orphans left, or an orphan is found with no
       possibly parents.
    7) Perform nonlinear refinement of the completed tree.
    '''

    start_time = time.time()

    # Start out by copying observed node set, as we'll be mutating their rule_k variables.
    observed_nodes = deepcopy(observed_nodes)

    candidate_intermediate_nodes = generate_candidate_intermediate_nodes(
        grammar, observed_nodes, max_recursion_depth=max_recursion_depth, verbose=verbose,
    )
    
    ## Rebuild starting from this partial, with a couple of restarts
    # in case one attempt fails.
    for attempt_k in range(max_attempts):
        # Remove child-to-rule assignments for all nodes whose parents we don't
        # know yet.
        for observed_node in observed_nodes:
            observed_node.rule_k = None
        for node in candidate_intermediate_nodes:
            node.rule_k = None
        tree_guess = SceneTree.make_from_observed_nodes(observed_nodes)
        # If one of the observed nodes is of root type, then don't
        # add a root node. Otherwise, go ahead and add the root node.
        root_node = None
        for node in tree_guess.nodes:
            if isinstance(node, grammar.root_node_type):
                root_node = node
        if root_node is None:
            root_node = grammar.root_node_type(tf=grammar.root_node_tf.detach())
            tree_guess.add_node(root_node)

        tree_guess = attempt_tree_repair_in_place(
            tree_guess, root_node, candidate_intermediate_nodes,
            max_iterations=max_iterations_per_attempt
        )
        score = tree_guess.score()
        if torch.isfinite(score):
            break
   
    if not torch.isfinite(score):
        logging.error("Failed to find feasible tree by greedy parsing.")

    end_time = time.time()
    if verbose:
        print("Found tree with score %f in %fs" % (score, end_time - start_time))
    return tree_guess, score

def infer_mle_tree_with_mip_from_proposals(
        grammar, observed_nodes, proposed_poses_by_unobserved_type,
        verbose=False, N_solutions=1, max_recursion_depth=10,
        min_ll_for_consideration=-100.):
    '''
        Set up a MIP to recover MLE parse trees, using the proposed
        intermediate node locations as options for how to place
        intermediate nodes.

        Uses the equivalent set abstraction to minimize the number of
        node-to-node score evaluations considered.

        Build a supertree for the grammar. For observed nodes, add

        For each node in the supertree,
        build a set of the actual poses that that node could take, and use
        a set of binary variables to choose among them; for the root and
        observed nodes, enforce 
        root, add binary activation variables for each node in the supertree and appropriate constraints
        and costs on parent/child activations implementing discrete
        probabilities (in the same way as MIP MAP parsing). Correspond
        the observed node types in the supertree to their corresponding
        observed nodes in the observed node set, forcing all observed
        nodes to be explained exactly once. 

        We can build a graph where each node is an observed
        or candidate intermediate node, and there is a directed
        edge for every legal rule, with a weight corresponding to its
        probability. We add one binary variable per edge indicating
        its activation.
         - Every observed node except the root needs exactly one active incoming
           edge. The root needs exactly zero. (It shouldn't have any incoming
           edges anyway by assumptions about construction of the grammar.)
           Unobserved nodes have outgoing edges iff they have an active
           incoming edge.
         - The score (and constraints on) a node's set of outgoing edges
           depends on the node type. This includes symmetry breaking
           where appropriate.
         - Maximize the total score.
    '''
    # Make local copy of node sets, since I'll be mutating the input
    # nodes a bit.
    observed_nodes = deepcopy(observed_nodes)
    assert len(set(observed_nodes)) == len(observed_nodes), "Duplicate observeds."

    # Build supertree for the grammar.
    super_tree = grammar.make_super_tree(max_recursion_depth=max_recursion_depth, detach=True)
    super_tree_root = super_tree.get_root()
    
    # Form equivalent sets.
    equivalent_sets = make_equivalent_sets(super_tree)

    prog = MathematicalProgram()

    for node in super_tree:
        # Bookkeeping variable of whether this node is active, which will be
        # constrained to be equal to the number of active incoming edges, or
        # 1 for the root node.
        node._active = prog.NewBinaryVariables(1, "%s_active" % node.name)[0]

    # Populate correspondences for observed nodes.
    for n in super_tree:
        if n.observed:
            n._possible_observations = []
            n._correspondences = []
            n._possible_tfs = []
    for node_k, observed_node in enumerate(observed_nodes):
        possible_sources = [n for n in super_tree if type(n) == type(observed_node)]
        if len(possible_sources) == 0:
            raise ValueError("Grammar invalid for observation: can't explain observed node ", observed_node)
        o_to_T_correspondences = prog.NewBinaryVariables(len(possible_sources), observed_node.__class__.__name__ + str(node_k) + "_sources")

        # Store these variables
        observed_node._o_to_T_correspondences = o_to_T_correspondences
        for k, n in enumerate(possible_sources):
            n._possible_observations.append(observed_node)
            n._correspondences.append(o_to_T_correspondences[k])
            n._possible_tfs.append(observed_node.tf)
        # Each observed node needs exactly one explaining input.
        prog.AddLinearEqualityConstraint(sum(o_to_T_correspondences) == 1)

    def get_possible_tfs_for_nonobserved_node(node):
        # For each non-observed node, collect its proposed poses.
        assert not node.observed
        if node is super_tree_root:
            return [grammar.root_node_tf]
        type_name = type(node).__name__
        if type_name not in proposed_poses_by_unobserved_type.keys():
            logging.warn("%s had no proposals." % type_name)
            return []
        possible_tfs = proposed_poses_by_unobserved_type[type_name]
        if len(possible_tfs) == 0:
            logging.warn("%s had zero proposals." % type_name)
        return possible_tfs

    # For each equivalent set, annotate the equivalent set with its possible
    # translation and rotation options, and activation variables indicating which
    # of those is the current chosen translation/rotation for the set.
    def collapse_close_entries(xs, *cs_srcs):
        new_xs = []
        cs_dsts = [[] for cs in cs_srcs]
        for i in range(len(xs)):
            found = False
            for j in range(len(new_xs)):
                if torch.allclose(xs[i], new_xs[j]):
                    found = True
                    for cs_dst, cs_src in zip(cs_dsts, cs_srcs):
                        cs_dst[j] += cs_src[i]
                    break
            if not found:
                new_xs.append(xs[i])
                for cs_src, cs_dst in zip(cs_srcs, cs_dsts):
                    cs_dst.append(cs_src[i])
        return [new_xs,] + cs_dsts

    for set_k, equivalent_set in enumerate(equivalent_sets):
        # Give every node a pointer to its set.
        for node in equivalent_set:
            node._equivalent_set = equivalent_set

        # For observed nodes, keep careful track of which
        # correspondences are leading to which pose
        # choice.
        obs_tfs = []
        obs_cs = []
        for node in equivalent_set:
            if node.observed:
                obs_tfs += node._possible_tfs
                obs_cs += node._correspondences
        # Reduce this to a map of poses, and expressions indicating
        # whether that pose is active.
        obs_tfs, obs_cs, obs_counts = collapse_close_entries(obs_tfs, obs_cs, np.ones(len(obs_cs)))
        if not equivalent_set.is_always_observable():
            # For the unobserved nodes, just collect the additional
            # pose choices available.
            unobs_tfs = []
            for node in equivalent_set:
                if not node.observed:
                    unobs_tfs += get_possible_tfs_for_nonobserved_node(node)
            unobs_tfs, = collapse_close_entries(unobs_tfs)
            # Create binary activations for activation those additional correspondences.
            unobs_cs = prog.NewBinaryVariables(len(unobs_tfs), "%d_unobs_corrs" % set_k)

            # Combine into the complete selection of ts.
            full_tfs, full_cs, full_counts = collapse_close_entries(
                obs_tfs + unobs_tfs,
                np.concatenate([obs_cs, unobs_cs]),
                np.concatenate([obs_counts, np.ones(len(unobs_tfs))])
            )
        else:
            full_tfs, full_cs, full_counts = obs_tfs, obs_cs, obs_counts

        # Constrain that exactly one entry in full_cs is allowed to be nonzero
        # by using an additional set of binaries constrained to be 1 when
        # the corresponding entry in full_cs is nonzero, and zero otherwise.
        equivalent_set.tf_correspondences = prog.NewBinaryVariables(len(full_tfs), "%d_t_corrs" % set_k)
        equivalent_set.tf_possibilities = full_tfs
        for tf_corr, cs_expression, count in zip(equivalent_set.tf_correspondences, full_cs, full_counts):
            prog.AddLinearConstraint(tf_corr >= cs_expression / count)
            prog.AddLinearConstraint(tf_corr <= cs_expression)
        # Only one of these correspondences, which correspond to distinct
        # poses, can be active simultaneously; and none are active if there
        # are no active nodes in the equivalent set.
        if len(equivalent_set.tf_correspondences) > 0:
            prog.AddLinearConstraint(sum(equivalent_set.tf_correspondences) <= 1)
            prog.AddLinearConstraint(sum(equivalent_set.tf_correspondences) <= sum([node._active for node in equivalent_set]))


    # Now at a node level, decide what will be active.
    for node in super_tree:
        # Nodes can't be active if their equiv set has no assigned
        # pose.
        prog.AddLinearConstraint(node._active <= sum(node._equivalent_set.tf_correspondences))

        if node is super_tree_root:
            prog.AddLinearEqualityConstraint(node._active == 1)
        elif node.observed:
            # Possible TFs are the full set of observed node tfs,
            # as populated in a previous loop. The observed node is
            # active iff it is corresponded to an observation.
            if len(node._correspondences) > 0:
                prog.AddLinearConstraint(sum(node._correspondences) == node._active)
            else:
                prog.AddLinearConstraint(node._active == 0)
        else:
            pass


    # For each parent/child pair in the supertree that crosses an equivalent
    # set boundary, choose among all possible edges weighted by the edge cost.
    for parent_node in super_tree:
        children = super_tree.get_children(parent_node)
        ## Get child rule list. Can't use get_children_and_rules
        # here since we're operating on a supertree, so the standard
        # scene tree logic for getting rules isn't correct.
        if isinstance(parent_node, RepeatingSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))


        #print("Processing ", parent_node.name)
        # Pick from every way of picking the pose of the parent + child,
        # weighted by the score of that edge.
        for child_node, rule in zip(children, rules):
            prog.AddLinearConstraint(child_node._active <= parent_node._active)
            if parent_node._equivalent_set != child_node._equivalent_set:
                # Build full set of possible parent/child R pairs
                # and their corresponding scores.
                activation_pairs = []
                scores = []
                #print("%s->%s has %dx%d pairs" % (parent_node.name, child_node.name, len(parent_node._equivalent_set.tf_correspondences), len(child_node._equivalent_set.tf_correspondences)))
                for parent_tf_active, parent_tf in zip(
                        parent_node._equivalent_set.tf_correspondences,
                        parent_node._equivalent_set.tf_possibilities):
                    for child_tf_active, child_tf in zip(
                            child_node._equivalent_set.tf_correspondences,
                            child_node._equivalent_set.tf_possibilities):
                        # Use supertree nodes as proxies for rule evaluation.
                        parent_node.tf = parent_tf
                        child_node.tf = child_tf
                        score = rule.score_child(parent_node, child_node)
                        if score < min_ll_for_consideration:
                            continue
                        activation_pairs.append((parent_tf_active, child_tf_active))
                        scores.append(score)
                if len(activation_pairs) == 0:
                    # No way to generate this child set, so this edge in the supertree
                    # will never be active.
                    prog.AddLinearEqualityConstraint(child_node._active == 0)
                    continue
                scores = np.array(scores)
                # Decide which edge (if any) is active.
                edge_actives = prog.NewBinaryVariables(len(activation_pairs), "%s-->%s_rule_choice" % (parent_node.name, child_node.name))
                # Only one edge can be active; edges activate their corresponding rule scores;
                # and the child is active iff an edge is on.
                prog.AddLinearConstraint(sum(edge_actives) == child_node._active)
                prog.AddLinearCost(-sum(edge_actives * scores))
                # The edge can only be active if the parent and child tf activation is on,
                # meaning the parent and child actually take that TF.
                for edge_active, (parent_tf_active, child_tf_active) in zip(edge_actives, activation_pairs):
                    # If either is off, then edge_active is forced to 0.
                    prog.AddLinearConstraint(edge_active <= (parent_tf_active + child_tf_active)/2.)


        # Apply constraints and costs over selecting this particular
        # child set to be active.
        child_actives = [c._active for c in children]

        # Constraints
        if isinstance(parent_node, RepeatingSetNode):
            # Geometric node child ordering to reduce duplicate parse: a child can
            # only be active if the previous child is also active.
            for k in range(len(child_actives) - 1):
                # child_{k+1} implies child_k
                # i.e. child_k >= child_{k+1}
                prog.AddLinearConstraint(child_actives[k] >= child_actives[k+1])
            # The geometric process we used (np.random.geometric) is only supported
            # on k=1, ..., so constrain that the # of active children must be
            # nonzero if this node is on.
            if len(child_actives) > 0:
                prog.AddLinearConstraint(sum(child_actives) >= parent_node._active)
            # Assert some arbitrary ordering to child poses: that they
            # ascend in their x coordinate. We can calculate the x coordinate
            # of children as a linear expression of the tf correspondence variable
            # and corresponding x coordinates. This constraint is deactivated by
            # a big M term if the second child in the comparison isn't active.
            child_xs = []
            # Get biggest possible x for big M deactivation
            max_child_x = max([max([tf[0, 3].detach().item() for tf in child._equivalent_set.tf_possibilities]) for child in children])
            # Accumulate expressions of child x
            for child in children:
                child_x = sum([
                    tf[0, 3].detach().item() * active for (tf, active) in zip(child._equivalent_set.tf_possibilities, child._equivalent_set.tf_correspondences)
                ])
                child_xs.append(child_x)
            for k in range(len(children) - 1):
                child_x = child_xs[k]
                next_child_x = child_xs[k+1]
                next_child = children[k+1]
                c = child_x <= next_child_x + max_child_x * (1. - next_child._active)
                if len(c.GetFreeVariables()) > 0:
                    # In some cases this formula simplifies to True, which breaks
                    # AddLinearConstraint.
                    prog.AddLinearConstraint(c)

        elif isinstance(parent_node, AndNode):
            # All children should be on if the parent node is on.
            for k in range(len(child_actives)):
                prog.AddLinearConstraint(child_actives[k] >= parent_node._active)
            # TODO(gizatt) Could do symmetry breaking here for children of
            # like type (e.g. singles-pairs constituency grammar).
        elif isinstance(parent_node, OrNode):
            # Exactly one child can be on if the parent is on.
            prog.AddLinearConstraint(sum(child_actives) == parent_node._active)
        elif isinstance(parent_node, (TerminalNode, IndependentSetNode)):
            # No additional constraint on which set of children should be active.
            pass
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))

        # Costs
        if isinstance(parent_node, (AndNode, TerminalNode)):
            pass
        elif isinstance(parent_node, OrNode):
            for p, child in zip(parent_node.rule_probs, children):
                prog.AddLinearCost(-np.log(p) * child._active)
        elif isinstance(parent_node, (OrNode, IndependentSetNode)):
            # Binary variables * log of probabilities.
            # Node inactive, active var off -> 0
            # Node active, active var on -> On score
            # Node active, active var off -> Off score
            # If parent is inactive, all children are inactive, so these probs go to zero.
            for p, child in zip(parent_node.rule_probs.detach().numpy(), children):
                prog.AddLinearCost(-(
                    np.log(p) * child._active + np.log(1 - p) * (1 - child._active) - np.log(1 - p) * (1 - parent_node._active)
                ))
        elif isinstance(parent_node, RepeatingSetNode):
            # Copy probabilities out from the "fake" geometric dist.
            count_probs = parent_node.rule_probs.detach().numpy()
            for child_k, child in enumerate(children):
                # Any given child being active removes the prob of the last #
                # of children being active, and adds the prob of this # of children
                # being active.
                if child_k == 0:
                    prog.AddLinearCost(-np.log(count_probs[child_k]) * child._active)
                else:
                    prog.AddLinearCost(
                        (np.log(count_probs[child_k-1]) -
                         np.log(count_probs[child_k])) 
                        * child._active
                    )
        else:
            raise ValueError("Unexpected node in cost assembly: ", type(parent_node))
        
    solver = GurobiSolver()
    options = SolverOptions()
    logfile = "/tmp/gurobi_%s.log" % datetime.now().strftime("%Y%m%dT%H%M%S")
    os.system("rm -f %s" % logfile)
    options.SetOption(solver.id(), "LogFile", logfile)
    options.SetOption(solver.id(), "MIPGap", 1E-3)
    if N_solutions > 1:
        options.SetOption(solver.id(), "PoolSolutions", N_solutions)
        options.SetOption(solver.id(), "PoolSearchMode", 2)

    result = solver.Solve(prog, None, options)
    # Hacky method getter because `num_suboptimal_solution()` was bound with () in its
    # method name. Should fix this upstream!
    actual_N_solutions = getattr(result, "num_suboptimal_solution()")()
    if actual_N_solutions < N_solutions:
        logging.warning("MIP got %d solutions, but requested %d. ", actual_N_solutions, N_solutions)
    if verbose:
        print("Optimization success?: ", result.is_success())
        print("Logfile: ")
        with open(logfile) as f:
            print(f.read())

    # Build tree for each solution
    out_trees = []
    for sol_k in range(actual_N_solutions):
        if verbose:
            print("Building tree for sol %d..." % sol_k)
        def get_sol(var):
            return result.GetSuboptimalSolution(var, sol_k)

        # Sanity-check observed nodes are explained properly.
        for observed_node in observed_nodes:
            if not np.isclose(np.sum(get_sol(observed_node._o_to_T_correspondences)), 1.):
                logging.error("Observed node %s not explained by MLE sol." % str(observed_node))

        optimized_tree = SceneTree()

        # Build tree top-down so we know parent is already in new tree.
        potential_node_queue = [(super_tree.get_root(), None)]
        while len(potential_node_queue) > 0:
            node, parent = potential_node_queue.pop(0)
            if get_sol(node._active) > 0.5:
                tf_actives = get_sol(node._equivalent_set.tf_correspondences)

                assert sum(tf_actives) == 1, "Active node had no active TF."
                tf = node._equivalent_set.tf_possibilities[np.argmax(tf_actives)]
                new_node = deepcopy(node)
                new_node.tf = tf
                optimized_tree.add_node(new_node)
                if parent is not None:
                    if verbose > 1:
                        print("Added %s--(%d)>%s" % (parent.name, new_node.rule_k, new_node.name))
                    optimized_tree.add_edge(parent, new_node)
                children = list(super_tree.successors(node))
                for child in children:
                    potential_node_queue.append((child, new_node))

        optim_score = torch.tensor(result.get_suboptimal_objective(sol_k))
        assert torch.isclose(optimized_tree.score(), -optim_score), "%f vs %f" % (optimized_tree.score(verbose=True), -optim_score)
        out_trees.append(optimized_tree)
    return out_trees


'''
###############################################################################
                          Mixed-integer parsing.

Parameterize the space of parse trees by forming a "super tree"
for the grammar. Create binary activation variables and continuous
pose variables for each node that the grammar could possibly produce.
Optimize those variables to form a maximum-probability tree that
explains the observed nodes.

Impose feasibility constraints on each parent/child pair:
- Child activation implies parent activation
- If the parent is an AND node and is active, all children are active.
- If the parent is an OR node and is active, exactly one child is active.
- Child pose in feasible region w.r.t parent node.

Impose additional symmetry breaking constraints:
- Prefer children activate in left-to-right order, and are assigned ascending poses in the x coordinate (symmetry breaking).

Add costs summing up activation of each child node.

###############################################################################
'''

def add_mle_tree_parsing_to_prog(
        prog, grammar, observed_nodes, max_recursion_depth=10,
        num_intervals_per_half_axis=2, max_scene_extent_in_any_dir=10.,
        verbose=False, use_random_rotation_offset=True):
    # The grammar should be pre-processed by `prepare_grammar_for_parsing`,
    # which creates decision variables or placeholders for the grammar parameters,
    # which we access from here.
    super_tree = grammar.make_super_tree(max_recursion_depth=max_recursion_depth, detach=True)
    root_node = super_tree.get_root()
    # Copy observed node set -- we'll be annotating the nodes with decision variables.
    observed_nodes = deepcopy(observed_nodes)

    # Support using a random rotation offset. For a random offset R_o, when a rotation matrix
    # encoded in decision variables R_dec is used, it's first (left) multiplied with R_o. Since many
    # optimal node poses may be axis-aligned (i.e. have zeros in the rotation matrix),
    # this means R_dec usually *won't* have zeros. Not having zeros in R_dec is important
    # to minimize the number of equivalent binary variable settings that represent the
    # same rotation; zeros in the rotation matrix are at the overlap of the piecewise
    # McCormick envelope setup used to approximate the rotation constraint.
    # (see sandbox/mip_rotation_constraint_solution_uniqueness).
    if use_random_rotation_offset:
        R_random_offset = UniformlyRandomRotationMatrix(RandomGenerator(42))
    else:
        R_random_offset = RotationMatrix(np.eye(3))

    # Every node gets a binary variable to indicate
    # whether it's active or node. This is equivalent to a
    # binary variable on the incoming edge for the node.
    active_vars = prog.NewBinaryVariables(len(super_tree.nodes), "nodes_active")
    for k, node in enumerate(super_tree.nodes):
        node.active = active_vars[k]

    if verbose:
        print("Activation vars allocated.")

    # For each observed node, add a binary variable for each possible
    # correspondence to a node in the observed set, where an active correspondence
    # forces the corresponded node to be the same position as the observed node.
    for n in super_tree:
        # (first prep some bookkeeping)
        n.possible_observations = []
    for node_k, observed_node in enumerate(observed_nodes):
        obs_tf = torch_tf_to_drake_tf(observed_node.tf)
        possible_sources = [n for n in super_tree if type(n) == type(observed_node)]
        if len(possible_sources) == 0:
            raise ValueError("Grammar invalid for observation: can't explain observed node ", observed_node)
        source_actives = prog.NewBinaryVariables(len(possible_sources), observed_node.__class__.__name__ + str(node_k) + "_sources")

        # Store these variables
        observed_node.source_actives = source_actives
        for k, n in enumerate(possible_sources):
            n.possible_observations.append((observed_node, source_actives[k], obs_tf))

        # Each observed node needs exactly one explaining input.
        prog.AddLinearEqualityConstraint(sum(source_actives) == 1)

    # Now for each observed node type in the supertree:
    #  - Ensure the node only observed ones actual observation.
    #  - If there are no observations of this type, this node can't be active.
    for node in super_tree:
        if node.observed:
            if len(node.possible_observations) > 0:
                sum_of_obs_vars = sum([obs_var for (_, obs_var, _) in node.possible_observations])
                prog.AddLinearConstraint(sum_of_obs_vars == node.active)
                prog.AddLinearConstraint(sum_of_obs_vars <= 1.) # Probably unnecessary
            else:
                # Never observed this type in the scene, so this node can't be active.
                prog.AddLinearConstraint(node.active == 0)


    # For translations and rotations separately, figure out which connected sets of nodes in the
    # supertree have translations [or rotations] that are constrained to be exactly equal.
    t_equivalent_sets, R_equivalent_sets = make_equivalent_sets_for_R_and_t(super_tree)

    # For each set, figure out if activation of any node in the set implies that
    # an observed node will be active. Record the result and a reference to the
    # corresponding set of observed nodes to the rest of the set.
    for t_equivalent_set in t_equivalent_sets:
        t_always_observed = t_equivalent_set.is_always_observable()
        t_observed_nodes = [node for node in t_equivalent_set if node.observed]
        for node in t_equivalent_set:
            node.t_always_observed = t_always_observed
            node.t_equivalent_to_observed_nodes = t_observed_nodes
    for R_equivalent_set in R_equivalent_sets:
        R_always_observed = R_equivalent_set.is_always_observable()
        R_observed_nodes = [node for node in R_equivalent_set if node.observed]
        for node in R_equivalent_set:
            node.R_always_observed = R_always_observed
            node.R_equivalent_to_observed_nodes = R_observed_nodes

    # For each set of nodes with equivalent translations [or rotations], create
    # decision variables set's translation [or rotation], and share it to all of the nodes
    # in the set. If it does contain an observed node, constrain the observation
    # to match the set translation/rotation when the observation correspondence is active.
    mip_rot_gen = MixedIntegerRotationConstraintGenerator(
        approach = MixedIntegerRotationConstraintGenerator.Approach.kBilinearMcCormick,
        num_intervals_per_half_axis=num_intervals_per_half_axis,
        interval_binning = IntervalBinning.kLinear # Logarithmic is a bit more efficient, but will break my rotation constraint binary variable sharing stuff.
    )
    for k, t_equivalent_set in enumerate(t_equivalent_sets):
        # For some convenience in expression forming, we'll always have some
        # auxiliary variables for this node set pose.
        t_optim = prog.NewContinuousVariables(3, "t_optim_%d" % k)
        # Put some reasonable bounds on the unknown t_optim.
        prog.AddBoundingBoxConstraint(-np.ones(3)*max_scene_extent_in_any_dir, np.ones(3)*max_scene_extent_in_any_dir, t_optim)
    
        proto_node = next(iter(t_equivalent_set))
        t_observed_nodes = proto_node.t_equivalent_to_observed_nodes
        for observed_node in t_observed_nodes:
            if len(observed_node.possible_observations) == 0:
                continue
            observed_t = sum([
                obs_var * obs_tf.translation()
                for (_, obs_var, obs_tf) in observed_node.possible_observations
            ])
            # If there are no active correspondences, allow the translation to vary freely.
            M = 2.*max_scene_extent_in_any_dir
            no_correspondences = 1. - sum([obs_var for (_, obs_var, _) in observed_node.possible_observations])
            for i in range(3):
                prog.AddLinearConstraint(t_optim[i] <= observed_t[i] + no_correspondences*M)
                prog.AddLinearConstraint(t_optim[i] >= observed_t[i] - no_correspondences*M)
        for node in t_equivalent_set:
            node.t_optim = t_optim.reshape(3)

    for k, R_equivalent_set in enumerate(R_equivalent_sets):
        R_optim_pre_offset = prog.NewContinuousVariables(3, 3, "R_optim_%d" % k)
        R_optim = R_optim_pre_offset.dot(R_random_offset.matrix())
        R_optim_mip_info = None

        proto_node = next(iter(R_equivalent_set))
        R_observed_nodes = proto_node.R_equivalent_to_observed_nodes
        R_always_observed = proto_node.R_always_observed
        for observed_node in R_observed_nodes:
            if len(observed_node.possible_observations) == 0:
                continue
            observed_R = sum([
                obs_var * obs_tf.rotation().matrix()
                for (_, obs_var, obs_tf) in observed_node.possible_observations
            ])
            # If there are no active correspondences, allow the rotation to vary freely.
            M = 2.
            no_correspondences = 1. - sum([obs_var for (_, obs_var, _) in observed_node.possible_observations])
            for i in range(3):
                for j in range(3):
                    prog.AddLinearConstraint(R_optim[i, j] <= observed_R[i, j] + M * no_correspondences)
                    prog.AddLinearConstraint(R_optim[i, j] >= observed_R[i, j] - M * no_correspondences)
        if not R_always_observed:
            # Add expensive rotation constraints if we can't guarantee R will
            # always correspond to some observed rotation.
            prog.AddBoundingBoxConstraint(-np.ones(9), np.ones(9), R_optim_pre_offset.flatten())
            # SO(3) constraint
            R_optim_mip_info = {
                "rot_gen": mip_rot_gen,
                "B": mip_rot_gen.AddToProgram(R_optim_pre_offset, prog).B_
            }
        for node in R_equivalent_set:
            node.R_optim = R_optim.reshape(3, 3)
            node.R_optim_pre_offset = R_optim_pre_offset.reshape(3, 3)
            node.R_optim_mip_info = R_optim_mip_info
            node.R_random_offset = R_random_offset
        
    if verbose:
        print("Continuous variables and SO(3) constraints allocated for all equivalence sets.")


    # For rules, annotate them with their parameter set for
    # convenience later. (This saves some effort, since the
    # node-type-to-rule mapping is a big trickier to work out in
    # a supertree context since we have to hand-construct child-to-rule
    # mappings for different node types.)
    for node in super_tree.nodes:
        param_vars_by_rule = grammar.rule_params_by_node_type_optim[node.__class__.__name__]
        for rule, (xyz_optim_params, rot_optim_params) in zip(node.rules, param_vars_by_rule):
            rule.xyz_optim_params = xyz_optim_params
            rule.rot_optim_params = rot_optim_params

    # Constrain root node to have the grammar's root position. If it's observed, don't bother,
    # as it should be in the observation set at exactly this position.
    # TODO(gizatt) This observation is *not* considered part of the equivalence set
    # logic above, as making the decision of "parent is observed *or* root" makes the
    # logic a lot messier. Should I add it, or just rely on most grammars having the root
    # be observed anyway?
    if not root_node.observed:
        root_tf = torch_tf_to_drake_tf(grammar.root_node_tf)
        t_target = root_tf.translation().reshape(3)
        R_target = root_tf.rotation().matrix().reshape(3, 3)
        for i in range(3):
            # These are redundant if the translation/rotation is functionally observed
            if len(root_node.t_equivalent_to_observed_nodes) == 0:
                prog.AddLinearEqualityConstraint(root_node.t_optim[i] == t_target[i])
            for j in range(3):
                if len(root_node.R_equivalent_to_observed_nodes) == 0:
                    prog.AddLinearEqualityConstraint(root_node.R_optim[i, j] == R_target[i, j])


    ## For each node in the super tree, add logical implications on the child activations
    # under the parent, depending on the parent type.
    # Root node must be active.
    prog.AddLinearConstraint(root_node.active == 1)
    for parent_node in super_tree:
        children = super_tree.get_children(parent_node)
        ## Get child rule list. Can't use get_children_and_rules
        # here since we're operating on a supertree, so the standard
        # scene tree logic for getting rules isn't correct.
        if isinstance(parent_node, RepeatingSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))

        child_actives = [c.active for c in children]
        parent_node.child_actives = child_actives

        # Children active requires parent active.
        for child_active in child_actives:
            prog.AddLinearConstraint(parent_node.active >= child_active)

        ## Parent/child activation relationships and symmetry breaking.
        if isinstance(parent_node, RepeatingSetNode):
            # Geometric node child ordering to reduce duplicate parse: a child can
            # only be active if the previous child is also active.
            for k in range(len(child_actives) - 1):
                # child_{k+1} implies child_k
                # i.e. child_k >= child_{k+1}
                prog.AddLinearConstraint(child_actives[k] >= child_actives[k+1])

            # Child supports should be ordered, to break symmetries.
            # TODO(gizatt) Does this apply to AND nodes? I don't think so, since
            # each node under an AND node has its own distribution. But I could conceivably
            # detect "equivalent" children and break symmetries there.
            for child, next_child in zip(children[:-1], children[1:]):
                c = next_child.t_optim[0] >= child.t_optim[0]
                if len(c.GetFreeVariables()) > 0:
                    # In some cases this formula simplifies to True, which breaks
                    # AddLinearConstraint.
                    prog.AddLinearConstraint(c)

            # The geometric process we used (np.random.geometric) is only supported
            # on k=1, ..., so constrain that the # of active children must be
            # nonzero if this node is on.
            if len(child_actives) > 0:
                prog.AddLinearConstraint(sum(child_actives) >= parent_node.active)

        elif isinstance(parent_node, AndNode):
            # All children should be on if the parent node is on.
            for k in range(len(child_actives)):
                prog.AddLinearConstraint(child_actives[k] >= parent_node.active)

        elif isinstance(parent_node, OrNode):
            # Exactly one child can be on if the parent is on.
            prog.AddLinearConstraint(sum(child_actives) == parent_node.active)

        elif isinstance(parent_node, (TerminalNode, IndependentSetNode)):
            # No additional constraint on which set of children should be active.
            pass
        
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))

    # Finally, build the objective.
    for parent_node in super_tree:
        ## p(child set | parent), which depends on the child activation variables.
        children = super_tree.get_children(parent_node)
        ## Get child rule list.
        if isinstance(parent_node, RepeatingSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))

        if isinstance(parent_node, (AndNode, TerminalNode)):
            pass
        elif isinstance(parent_node, OrNode):
            for p, child in zip(parent_node.rule_probs, children):
                prog.AddLinearCost(-np.log(p) * child.active)
        elif isinstance(parent_node, IndependentSetNode):
            # Binary variables * log of probabilities.
            # Node inactive, active var off -> 0
            # Node active, active var on -> On score
            # Node active, active var off -> Off score
            # If parent is inactive, all children are inactive, so these probs go to zero.
            for p, child in zip(parent_node.rule_probs.detach().numpy(), children):
                prog.AddLinearCost(-(
                    np.log(p) * child.active + np.log(1 - p) * (1 - child.active) - np.log(1 - p) * (1 - parent_node.active)
                ))
        elif isinstance(parent_node, RepeatingSetNode):
            # Copy probabilities out from the "fake" geometric dist.
            count_probs = parent_node.rule_probs.detach().numpy()
            for child_k, child in enumerate(children):
                # Any given child being active removes the prob of the last #
                # of children being active, and adds the prob of this # of children
                # being active.
                if child_k == 0:
                    prog.AddLinearCost(-np.log(count_probs[child_k]) * child.active)
                else:
                    prog.AddLinearCost(
                        (np.log(count_probs[child_k-1]) -
                         np.log(count_probs[child_k])) 
                        * child.active
                    )
        else:
            raise ValueError("Unexpected node in cost assembly: ", type(parent_node))

        ## p(child tf | parent tf) for each child.
        for rule, child_node in zip(rules, children):
            # Can't apply SamePositionRule/SameRotationRule again,
            # as due to our grouping into equivalence classes, this would
            # add trivial constraints (variable == variable) that cause Drake
            # to throw errors. All other constraints should be fair game.
            if not isinstance(rule.xyz_rule, SamePositionRule):
                rule.xyz_rule.encode_constraint(
                    prog, rule.xyz_optim_params, parent_node, child_node, max_scene_extent_in_any_dir
                )
            if not isinstance(rule.rotation_rule, SameRotationRule):
                rule.rotation_rule.encode_constraint(
                    prog, rule.rot_optim_params, parent_node, child_node, max_scene_extent_in_any_dir
                )
            rule.encode_cost(
                prog, rule.xyz_optim_params, rule.rot_optim_params, child_node.active, parent_node, child_node, max_scene_extent_in_any_dir
            )

    return super_tree, observed_nodes, R_random_offset

# Return type of infer_mle_tree_with_mip
TreeInferenceResults = namedtuple("TreeInferenceResults", ["solver", "optim_result", "super_tree", "observed_nodes", "R_random_offset"])
# TODO(gizatt) Remote max_scene_extent_in_any_dir and calculate from grammar.
def infer_mle_tree_with_mip(grammar, observed_nodes, max_recursion_depth=10, solver="gurobi", verbose=False,
                            num_intervals_per_half_axis=2, max_scene_extent_in_any_dir=10.,
                            N_solutions=1, use_random_rotation_offset=True):
    ''' Given a grammar and an observed node set, find the MLE tree induced
    by that grammar (up to a maximum recursion depth) that reproduces the
    observed node set, or report that it's infeasible. '''
    start_time = time.time()
    if verbose:
        print("Starting setup.")

    prog = MathematicalProgram()

    grammar = prepare_grammar_for_parsing(prog, grammar, optimize_parameters=False)
    super_tree, observed_nodes, R_random_offset = add_mle_tree_parsing_to_prog(
        prog, grammar, observed_nodes, max_recursion_depth=max_recursion_depth,
        verbose=verbose, num_intervals_per_half_axis=num_intervals_per_half_axis,
        max_scene_extent_in_any_dir=max_scene_extent_in_any_dir,
        use_random_rotation_offset=use_random_rotation_offset
    )
    
    setup_time = time.time()
    if verbose:
        print("Setup time: ", setup_time - start_time)
        print("Num vars: ", prog.num_vars())
        print("Num constraints: ", sum([c.evaluator().num_constraints() for c in prog.GetAllConstraints()]))
        sys.stdout.flush()

    if solver == "gurobi":
        solver = GurobiSolver()
        options = SolverOptions()
        logfile = "/tmp/gurobi_%s.log" % datetime.now().strftime("%Y%m%dT%H%M%S")
        os.system("rm -f %s" % logfile)
        options.SetOption(solver.id(), "LogFile", logfile)
        options.SetOption(solver.id(), "MIPGap", 1E-3)
        if N_solutions > 1:
            options.SetOption(solver.id(), "PoolSolutions", N_solutions)
            options.SetOption(solver.id(), "PoolSearchMode", 2)

        result = solver.Solve(prog, None, options)
        # Hacky method getter because `num_suboptimal_solution()` was bound with () in its
        # method name. Should fix this upstream!
        actual_N_solutions = getattr(result, "num_suboptimal_solution()")()
        if actual_N_solutions < N_solutions:
            logging.warning("MIP got %d solutions, but requested %d. ", actual_N_solutions, N_solutions)
        if verbose:
            print("Optimization success?: ", result.is_success())
            print("Logfile: ")
            with open(logfile) as f:
                print(f.read())

    elif solver == "branch_and_bound":
        # Mostly for testing; supplies a different result type.
        solver = MixedIntegerBranchAndBound(prog, ClpSolver().solver_id())
        result = solver.Solve()
        if verbose:
            print("Optimization success?: ", result == SolutionResult.kSolutionFound)
        # Return solver, since result is just a SolutionResult object.
        # (Why does the MIBNB solver not use SolverInterface?)
        result = [solver, result == SolutionResult.kSolutionFound]
    else:
        raise ValueError("Unknown solver: ", solver)

    solve_time = time.time() 
    if verbose:
            print("Solve time: ", solve_time-setup_time)
            print("Total time: ", solve_time - start_time)

    # Manually prune these out; they're hard to clean up for pickling, and not necessary
    # for anyone else to consume.
    # TODO(gizatt): Need to cleanly contain all extra optimization variables in a container
    # so I can prune them before pickling.
    for node in super_tree:
        del node.R_equivalent_to_observed_nodes
        del node.t_equivalent_to_observed_nodes
        del node.R_optim_mip_info

    return TreeInferenceResults(solver, result, super_tree, observed_nodes, R_random_offset)


def get_optimized_tree_from_mip_results(inference_results, assert_on_failure=False, solution_k=0):
    ''' From the specified inference results, extract the k^th suboptimal scene tree. For k=0,
    returns the optimal solution. '''
    
    #TODO packing of TreeInferenceResults should handle this complexity
    if isinstance(inference_results.solver,  MixedIntegerBranchAndBound):
        optim_result, success = (
            inference_results.solver,
            inference_results.optim_result
        )
        assert solution_k == 0, "MIBnB doesn't support suboptimal sols."
        def get_sol(var):
            return optim_result.GetSolution(var)

    elif isinstance(inference_results.solver, GurobiSolver):
        optim_result, success = (
            inference_results.optim_result,
            inference_results.optim_result.is_success()
        )
        def get_sol(var):
            return optim_result.GetSuboptimalSolution(var, solution_k)

    else:
        raise ValueError(inference_results.solver)

    super_tree = inference_results.super_tree
    observed_nodes = inference_results.observed_nodes

    if not success:
        logging.error("MIP structure finding unsuccessful.")
        return None

    # Sanity-check observed nodes are explained properly.
    for observed_node in observed_nodes:
        if not np.isclose(np.sum(get_sol(observed_node.source_actives)), 1.):
            if assert_on_failure:
                raise ValueError("observed node %s not explained by MLE sol." % str(observed_node))
            else:
                logging.error("Observed node %s not explained by MLE sol." % str(observed_node))

    optimized_tree = SceneTree()

    # Reconstruct what the optimization thinks our score should be.
    # (Debugging code)
    '''
    total_active_score = 0.
    for parent_node in super_tree:
        if get_sol(parent_node.active) < 0.5:
            continue
        children = super_tree.get_children(parent_node)
        ## Get child rule list.
        if isinstance(parent_node, RepeatingSetNode):
            rules = [parent_node.rule for k in range(len(children))]
        elif isinstance(parent_node, (AndNode, OrNode, IndependentSetNode)):
            rules = parent_node.rules
        elif isinstance(parent_node, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent_node))
        child_set_score = parent_node.score_child_set([child for child in children if get_sol(child.active) > 0.5]).item()
        print("%s (active %f): child set score %f" % (parent_node.name, get_sol(parent_node.active), child_set_score))
        if get_sol(parent_node.active) > 0.5:
            total_active_score += child_set_score
        
        proxy_parent = deepcopy(parent_node)
        t_sol = get_sol(parent_node.t_optim)
        R_sol = RotationMatrix(
            get_sol(parent_node.R_optim_pre_offset).dot(inference_results.R_random_offset.matrix())
        )
        new_tf = drake_tf_to_torch_tf(RigidTransform(
            p=t_sol, R=R_sol
        ))
        proxy_parent.tf = new_tf

        for child, rule in zip(children, rules):
            if get_sol(child.active) < 0.5:
                continue
            proxy_child = deepcopy(child)
            t_sol = get_sol(child.t_optim)
            R_sol = RotationMatrix(
                get_sol(child.R_optim_pre_offset).dot(inference_results.R_random_offset.matrix())
            )
            new_tf = drake_tf_to_torch_tf(RigidTransform(
                p=t_sol, R=R_sol
            ))
            proxy_child.tf = new_tf
            child_score = rule.score_child(proxy_parent, proxy_child, verbose=1).item()
            print("%s:%s (active %s): child %f" % (proxy_parent.name, proxy_child.name, get_sol(child.active), child_score))
            if get_sol(parent_node.active) > 0.5 and get_sol(child.active) > 0.5:
                total_active_score += child_score
    print("Total active score: ", total_active_score)
    '''
    
    # Build tree top-down so we know parent is already in new tree.
    potential_node_queue = [(super_tree.get_root(), None)]
    while len(potential_node_queue) > 0:
        node, parent = potential_node_queue.pop(0)
        if get_sol(node.active) > 0.5:
            t_sol = get_sol(node.t_optim)
            R_sol = RotationMatrix(
                    get_sol(node.R_optim_pre_offset).dot(
                        inference_results.R_random_offset.matrix()
                    )
            )
            new_tf = drake_tf_to_torch_tf(RigidTransform(
                p=t_sol, R=R_sol
            ))
            new_node = deepcopy(node)
            new_node.tf = new_tf
            optimized_tree.add_node(new_node)
            if parent is not None:
                optimized_tree.add_edge(parent, new_node)
            children = list(super_tree.successors(node))
            for child in children:
                potential_node_queue.append((child, new_node))

    return optimized_tree

def get_optimized_trees_from_mip_results(inference_results, assert_on_failure=False):
    if isinstance(inference_results.solver, MixedIntegerBranchAndBound):
        return [get_optimized_tree_from_mip_results(inference_results, assert_on_failure)]
    # Hacky method getter because `num_suboptimal_solution()` was bound with () in its
    # method name. Should fix this upstream!
    N_solutions = getattr(inference_results.optim_result, "num_suboptimal_solution()")()
    assert N_solutions >= 1
    return [get_optimized_tree_from_mip_results(inference_results, assert_on_failure, k) for k in range(N_solutions)]



'''
###############################################################################
###                    Nonlinear refinement of a parse tree.
###
###
###############################################################################
'''

TreeRefinementResults = namedtuple("TreeRefinementResults", ["optim_result", "refined_tree", "unrefined_tree"])
def optimize_scene_tree_with_nlp(grammar, scene_tree, initial_guess_tree=None, objective="mle", verbose=False, max_scene_extent_in_any_dir=10.):
    ''' Given a scene tree, set up a nonlinear optimization:
    1) Keeps the tree structure the same, but tweaks non-observed,
        non-root node poses.
    2) Optimizes for feasibility (of relative node poses).
    3) Allows choice of objective: maximum tree score ("mle"), or closeness to
        current tree configuration ("projection")
    4) Uses the scene tree's current (possibly infeasible) configuration
        as the initial guess.
    '''
    eps = 1E-4 # Need relatively loose epsilon, or NLP gets stuck.
    start_time = time.time()
    prog = MathematicalProgram()
    grammar = prepare_grammar_for_parsing(prog, grammar, optimize_parameters=False)

    if initial_guess_tree is None:
        initial_guess_tree = scene_tree

    # Add pose decision variables with constraints.
    root_node = scene_tree.get_root()
    for k, (node, hint_node) in enumerate(zip(scene_tree.nodes, initial_guess_tree)):
        # Declare decision variables for pose and seed them from
        # the input tree's poses.
        node.R_optim = prog.NewContinuousVariables(3, 3, "R_%d" % k)
        prog.SetInitialGuess(node.R_optim, hint_node.rotation.detach().cpu().numpy())
        node.t_optim = prog.NewContinuousVariables(3, "t_%d" % k)
        prog.SetInitialGuess(node.t_optim, hint_node.translation.detach().cpu().numpy())

        # If it's an observed node or the root node, constrain the pose
        # to not change, and that's it.
        if node.observed or node is root_node:
            # TODO(gizatt) Rip this apart and figure out why I need such loose feasibility
            # tolerances.
            R_target = node.rotation.cpu().detach().numpy()
            prog.AddBoundingBoxConstraint(R_target.flatten()-eps, R_target.flatten()+eps, node.R_optim.flatten())
            t_target = node.translation.cpu().detach().numpy()
            prog.AddBoundingBoxConstraint(t_target, t_target, node.t_optim)
        else:
            # Otherwise, constrain the pose to be a legal and good pose.
            # R.' R = I
            RtR = node.R_optim.T.dot(node.R_optim)
            I = np.eye(3)
            for i in range(3):
                for j in range(3):
                    prog.AddConstraint(RtR[i, j] >= I[i, j] - eps)
                    prog.AddConstraint(RtR[i, j] <= I[i, j] + eps)
            # det(R) = +1; using cross product form and expressing it loosely, just to
            # keep SNOPT from falsely returning a flip-and-rotation as a legitimate
            # solution. Idea here is that the first two columns of the rotation
            # are the X and Y axes of the new coordinate; the Z axis of that right-handed
            # coordinate system should be the same direction as the last column.
            z_dir = np.cross(node.R_optim[:, 0], node.R_optim[:, 1])
            prog.AddConstraint(np.dot(z_dir, node.R_optim[:, 2]) >= 0)
            # Strong bounding box on rotation matrix elements.
            #prog.AddBoundingBoxConstraint(-np.ones(9), np.ones(9), node.R_optim.flatten())
            # Translation
            # Add really loose bbox constraint, to keep SNOPT from running away.
            # TODO(gizatt) Update these to use scene tree production bounds, if I
            # ever get around to adding that.
            prog.AddBoundingBoxConstraint(
                -np.ones(3)*max_scene_extent_in_any_dir,
                np.ones(3)*max_scene_extent_in_any_dir,
            node.t_optim)

        
    # Constraint parent/child relationships.
    for parent_node in scene_tree.nodes:
        children, rules = scene_tree.get_children_and_rules(parent_node)

        ## Child location constraints relative to parent.
        optim_params_by_rule = grammar.rule_params_by_node_type_optim[type(parent_node).__name__]
        for rule, child_node, (xyz_optim_params, rot_optim_params) in zip(rules, children, optim_params_by_rule):
            rule.encode_constraint(
                prog, xyz_optim_params, rot_optim_params, parent_node, child_node, max_scene_extent_in_any_dir
            )

    def penalize_rotation_error(R_goal, R_optim):
        # We don't always have perfect rotations, so instead of using
        # arccos((tr(R_diff) - 1) / 2) (the angular distance),
        # I'll use the distance chord distance of a few vectors rotated
        # by the rotation difference, which goes to zero if the
        # rotations are the same. I use multiple vectors so the rotation
        # axis is the same as a vector, we still get signal.
        # TODO(gizatt) Would just quadratic error between desired + actual R
        # work just as well but be computationally easier?
        R_diff = R_goal.T.dot(R_optim)
        for k in range(3):
            vec = np.zeros(3)
            vec[k] = 1.
            vec_rot = R_diff.dot(vec)
            prog.AddCost((vec_rot - vec).T.dot(vec_rot - vec))

    def penalize_pose_error(pose_goal, t_optim, R_optim):
        prog.AddQuadraticErrorCost(np.eye(3), pose_goal.translation(), t_optim)
        penalize_rotation_error(pose_goal.rotation().matrix(), R_optim)

    if objective == "mle":
        # Add costs for MLE tree estimate
        for parent_node in scene_tree.nodes:
            children, rules = scene_tree.get_children_and_rules(parent_node)
            ## Child location costs relative to parent.
            optim_params_by_rule = grammar.rule_params_by_node_type_optim[type(parent_node).__name__]
            for rule, child_node, (xyz_optim_params, rot_optim_params) in zip(rules, children, optim_params_by_rule):
                if verbose:
                    print(rule.xyz_rule, rule.rotation_rule)
                rule.encode_cost(
                    prog, xyz_optim_params, rot_optim_params, True, parent_node, child_node, max_scene_extent_in_any_dir
                )

    elif objective == "projection":
        # Try to get optimized tree as close as possible to current config.
        for node in scene_tree.nodes:
            if not node.observed and node is not root_node:
                pose_goal = RigidTransform(
                    p=node.translation.cpu().detach().numpy(),
                    R=RotationMatrix(node.rotation.cpu().detach().numpy())
                )
                penalize_pose_error(pose_goal, node.t_optim, node.R_optim)


    else:
        raise ValueError("Unknown objective spec \"%s\"" % objective)

    ## Solve
    setup_time = time.time()
    solver = SnoptSolver()
    options = SolverOptions()
    logfile = "/tmp/snopt_%s.log" % datetime.now().strftime("%Y%m%dT%H%M%S")
    os.system("rm -f %s" % logfile)
    options.SetOption(solver.id(), "Print file", logfile)
    options.SetOption(solver.id(), "Major feasibility tolerance", eps)
    result = solver.Solve(prog, None, options)
    solve_time = time.time()

    if verbose:
        print("Success?: ", result.is_success())
        print("Logfile: ")
        with open(logfile) as f:
            print(f.read())
        print("Solve time: ", solve_time-setup_time)
        print("Total time: ", solve_time - start_time)
        print("Optimal cost: ", result.get_optimal_cost())

    out_tree = deepcopy(scene_tree)
    if result.is_success():
        # Copy results into scene tree node positions
        for out_node, orig_node in zip(out_tree, scene_tree):
            if orig_node is root_node or orig_node.observed:
                # Don't mutate the fixed nodes
                continue
            out_node.translation = torch.tensor(result.GetSolution(orig_node.t_optim))
            out_node.rotation = torch.tensor(result.GetSolution(orig_node.R_optim))
    else:
        logging.warning("Nonlinear refinement unsuccessful.")
    logging.info("Done")
    return TreeRefinementResults(result, out_tree, scene_tree)
