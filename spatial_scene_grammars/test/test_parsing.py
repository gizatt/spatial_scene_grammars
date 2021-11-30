import pytest

import torch

from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.parsing import *

from .grammar import *

from pydrake.all import SnoptSolver

torch.set_default_tensor_type(torch.DoubleTensor)

@pytest.fixture(params=range(3))
def set_seed(request):
    torch.manual_seed(request.param)

def assert_feasible(optimized_tree):
    # Assert optimized tree structure is feasible.
    score = optimized_tree.score(verbose=True)
    assert torch.isfinite(score), "Infeasible recovered score: %f" % score
    for opt_node in optimized_tree.nodes:
        # Expect failure due to imprecise constraint, but how close?
        R = opt_node.rotation
        t = opt_node.translation
        RtR = torch.matmul(R, R.transpose(0, 1))
        assert torch.allclose(RtR, torch.eye(3), rtol=1E-3, atol=1E-3), RtR
        det = torch.det(R)
        assert torch.isclose(det, torch.tensor(1.), rtol=1E-3, atol=1E-4), det

def assert_explains_observeds(observed_tree, optimized_tree):
    # Assert optimized tree structure explains observed nodes.
    for obs_node in observed_tree.get_observed_nodes():
        closest_dist = torch.tensor(1.)
        for matching_node in optimized_tree.find_nodes_by_type(type(obs_node)):
            R_dist = (matching_node.rotation - obs_node.rotation).abs().sum()
            t_dist = (matching_node.translation - obs_node.translation).abs().sum()
            closest_dist = min(closest_dist, R_dist + t_dist)
        assert torch.isclose(closest_dist, torch.zeros(1), atol=1E-4, rtol=1E-4), "Didn't match observation well enough: err %f" % closest_dist

# Parse a simple infeasible scene, and show that it's infeasible.
def test_parsing_infeasible():
    grammar = SpatialSceneGrammar(
        root_node_type = NodeG,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree(detach=True)
    observed_nodes = tree.get_observed_nodes()
    assert len(observed_nodes) == 1

    # Parsing with no observeds is going to be a problem,
    # since this grammar *must* produce an observation.
    inference_results = infer_mle_tree_with_mip(
        grammar, [], verbose=True,
        solver="branch_and_bound"
    )
    assert not inference_results.optim_result[1], "This scene should not be feasible."

# Try to parse a trivial subset of our grammar with a
# simple solver. (The solver is very slow with additional
# complexity.)
@pytest.mark.skipif(not SnoptSolver().available(),
                    reason='This test relies on SNOPT.')
def test_parsing_simple(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeD,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree(detach=True)
    observed_nodes = tree.get_observed_nodes()
    print(tree, observed_nodes)

    start_time = time.time()
    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True,
        solver="branch_and_bound"
    )
    elapsed = time.time() - start_time
    print("Parsing took %f secs." % elapsed)
    assert inference_results.optim_result[1]

    mip_optimized_trees = get_optimized_trees_from_mip_results(inference_results)
    assert len(mip_optimized_trees) == 1
    mip_optimized_tree = mip_optimized_trees[0]
    assert_explains_observeds(tree, mip_optimized_tree), "MIP parsing failed."

    optim_score = torch.tensor(inference_results.optim_result[0].GetOptimalCost())
    assert torch.allclose(mip_optimized_tree.score(), -optim_score), "%f vs %f" % (mip_optimized_tree.score(), -optim_score)
    
    start_time = time.time()
    refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert refinement_results.optim_result.is_success()
    refined_tree = refinement_results.refined_tree
    assert_explains_observeds(tree, refined_tree), "Refinement failed."
    assert_feasible(refined_tree)


## Try to parse samples from our very simple (but
# still nontrivial, due to rotations and hidden
# nodes) grammar.
@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing_complex(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree(detach=True)
    observed_nodes = tree.get_observed_nodes()

    start_time = time.time()
    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True, N_solutions=2
    )
    elapsed = time.time() - start_time
    print("Parsing took %f secs." % elapsed)
    assert inference_results.optim_result.is_success(), "MIP parsing failed."

    mip_optimized_trees = get_optimized_trees_from_mip_results(inference_results)
    mip_optimized_tree = mip_optimized_trees[0]

    assert_explains_observeds(tree, mip_optimized_tree)

    start_time = time.time()
    for node in mip_optimized_tree:
        print(node, node.tf)
    refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, objective="mle", verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert refinement_results.optim_result.is_success(), "Refinement failed."
    refined_tree = refinement_results.refined_tree
    assert_explains_observeds(tree, refined_tree)
    assert_feasible(refined_tree)

    # Check out if get-tree-as-close-to-this-one works too.
    tree = grammar.sample_tree(detach=True)
    # Perturb node C to a known feasible location.
    # It's always safe to put C anywhere in the unit box, since its spawning
    # rule puts it there, and it places all of its children there.
    C = tree.find_nodes_by_type(NodeC)[0]
    C_translation_goal = torch.tensor([0.1, 0.2, 0.3])
    C.translation = C_translation_goal
    projection_results = optimize_scene_tree_with_nlp(grammar, tree, objective="projection", verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert projection_results.optim_result.is_success(), "Refinement failed."
    refined_tree = projection_results.refined_tree
    C_refined = refined_tree.find_nodes_by_type(NodeC)[0]
    assert_explains_observeds(tree, refined_tree)
    assert_feasible(refined_tree)
    assert(torch.allclose(C_refined.translation, C_translation_goal))

    # This C *isn't* feasible; the closest feasible should be
    # within unit box.
    C = tree.find_nodes_by_type(NodeC)[0]
    C_translation_goal = torch.tensor([-0.1, 0.2, 0.3])
    C_expected = torch.tensor([0.0, 0.2, 0.3])
    C.translation = C_translation_goal
    projection_results = optimize_scene_tree_with_nlp(grammar, tree, objective="projection", verbose=True)
    elapsed = time.time() - start_time
    print("Refinement took %f secs." % elapsed)
    assert projection_results.optim_result.is_success(), "Refinement failed."
    refined_tree = projection_results.refined_tree
    C_refined = refined_tree.find_nodes_by_type(NodeC)[0]
    assert_explains_observeds(tree, refined_tree)
    assert_feasible(refined_tree)
    assert torch.allclose(C_refined.translation, C_expected), "%s vs %s" % (C_refined.translation, C_expected)


@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing_n_solutions():
    # Make a custom trivial grammar A->B(hidden)->C

    class C(TerminalNode):
        def __init__(self, tf):
            super().__init__(observed=True, physics_geometry_info=None, tf=tf)

    class B(AndNode):
        def __init__(self, tf):
            super().__init__(observed=False, physics_geometry_info=None, tf=tf)
        @classmethod
        def generate_rules(cls):
            return [ProductionRule(child_type=C, xyz_rule=SamePositionRule(), rotation_rule=UnconstrainedRotationRule())]

    class A(AndNode):
        def __init__(self, tf):
            super().__init__(observed=False, physics_geometry_info=None, tf=tf)
        @classmethod
        def generate_rules(cls):
            return [ProductionRule(child_type=B, xyz_rule=SamePositionRule(), rotation_rule=UnconstrainedRotationRule())]

    grammar = SpatialSceneGrammar(
        root_node_type = A,
        root_node_tf = torch.eye(4)
    )
    observed_nodes = [C(torch.eye(4))]

    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True, N_solutions=20,
        use_random_rotation_offset=False
    )
    assert inference_results.optim_result.is_success(), "MIP parsing failed."

    mip_optimized_trees = get_optimized_trees_from_mip_results(inference_results)

    # Should find a diversity of unconstrained solutions.
    assert len(mip_optimized_trees) == 20

all_rules = [
    WorldFrameBBoxRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)*3.),
    WorldFrameBBoxOffsetRule.from_bounds(lb=torch.zeros(3), ub=torch.ones(3)*5.),
    WorldFrameGaussianOffsetRule(mean=torch.zeros(3), variance=torch.ones(3)),
    ParentFrameGaussianOffsetRule(mean=torch.zeros(3), variance=torch.ones(3)),
    WorldFramePlanarGaussianOffsetRule(
        mean=torch.zeros(2), variance=torch.ones(2), plane_transform=RigidTransform(p=np.array([1., 2., 3.]), rpy=RollPitchYaw(1., 2., 3.))),
    UnconstrainedRotationRule(),
    UniformBoundedRevoluteJointRule.from_bounds(axis=torch.tensor([0., 1., 0.]), lb=-np.pi/2., ub=np.pi/2.),
    WorldFrameBinghamRotationRule(M=torch.eye(4), Z=torch.tensor([-1., -1., -1., 0.])),
    ParentFrameBinghamRotationRule(M=torch.eye(4), Z=torch.tensor([-1., -1., -1., 0.]))
]
@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
@pytest.mark.parametrize("rule", all_rules, ids=[type(rule).__name__ for rule in all_rules])
@pytest.mark.parametrize("parent_observed", [False, True])
@pytest.mark.parametrize("child_observed", [False, True])
def test_parsing_rules_in_each_observed_case(set_seed, rule, parent_observed, child_observed):
    # Construct a trivial grammar in which the only uncertainty is the pose of an
    # intermediate node. Experiment with each combination of parent/child of that
    # rule being observed and unobserved, making sure each case works as expected.

    # Case 1: A -> B -> C
    #              B -> C by the rule we're testing.
    #              B is observed or unobserved.
    class C(TerminalNode):
        def __init__(self, tf):
            super().__init__(observed=child_observed, physics_geometry_info=None, tf=tf)

    if isinstance(rule, RotationProductionRule):
        xyz_rule = SamePositionRule()
        rotation_rule = rule
    else:
        xyz_rule = rule
        rotation_rule = SameRotationRule()

    class B(AndNode):
        def __init__(self, tf):
            super().__init__(observed=parent_observed, physics_geometry_info=None, tf=tf)
        @classmethod
        def generate_rules(cls):
            return [ProductionRule(child_type=C, xyz_rule=xyz_rule, rotation_rule=rotation_rule)]

    class A(AndNode):
        def __init__(self, tf):
            super().__init__(observed=False, physics_geometry_info=None, tf=tf)
        @classmethod
        def generate_rules(cls):
            return [ProductionRule(child_type=B, xyz_rule=SamePositionRule(), rotation_rule=SameRotationRule())]

    grammar = SpatialSceneGrammar(
        root_node_type = A,
        root_node_tf = torch.eye(4)
    )
    observed_nodes = grammar.sample_tree(detach=True).get_observed_nodes()

    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True, N_solutions=3,
        use_random_rotation_offset=True
    )
    assert inference_results.optim_result.is_success(), "MIP parsing failed."

    mip_optimized_trees = get_optimized_trees_from_mip_results(inference_results)


@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None,
                    reason='This test relies on Gurobi.')
def test_proposal_parsing(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    tree = grammar.sample_tree(detach=True)
    observed_nodes = tree.get_observed_nodes()

    start_time = time.time()
    candidate_intermediate_nodes = generate_candidate_intermediate_nodes(
        grammar, observed_nodes, max_recursion_depth=10, verbose=True
    )
    parse_trees = infer_mle_tree_with_mip_from_proposals(
        grammar, observed_nodes, candidate_intermediate_nodes, verbose=True, N_solutions=1
    )
    elapsed = time.time() - start_time
    print("Parsing took %f secs." % elapsed)
    assert len(parse_trees) >= 1, "MIP parsing failed."

    for parse_tree in parse_trees:
        assert_explains_observeds(tree, parse_tree)