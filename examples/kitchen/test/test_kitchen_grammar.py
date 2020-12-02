import unittest
from unittest import TestLoader

from pydrake.all import (BodyIndex, FrameIndex, ModelInstanceIndex)
import numpy as np

import torch
import pyro

from scene_grammar.src.nodes import *
from scene_grammar.src.rules import *
from scene_grammar.src.tree import *
from scene_grammar.src.transform_utils import *
from scene_grammar.src.visualization import *
from scene_grammar.src.drake_interop import *
from scene_grammar.src.serialization import *

from scene_grammar.examples.kitchen.grammar_room_layout import *

torch.set_default_tensor_type(torch.DoubleTensor)
pyro.enable_validation(True)


class FixedSeedKitchenGrammarTests(unittest.TestCase):
    fixed_seed = 0
    @classmethod
    def setUpClass(cls):
        pyro.clear_param_store()
        torch.manual_seed(cls.fixed_seed)

        model = lambda: ParseTree.generate_from_root_type(
            root_node_type=Kitchen,
            tf=torch.eye(4),
            name="kitchen"
        )
        cls.trace = pyro.poutine.trace(model).get_trace()
        cls.scene_tree = cls.trace.nodes["_RETURN"]["value"]

    def test_finite_logprob(self):
        self.assertTrue(torch.isfinite(self.trace.log_prob_sum()))

    def test_simulation_setup(self):
        # Simulate the resulting scene.
        # May not expect this to work if the scene is generated in an
        # infeasible config... but short sim horizons seem not to error out,
        # so at least it checks the construction machinery and that Drake doesn't
        # hate the model on a fundamental level.
        simulate_scene_tree(self.scene_tree, 0.001, timestep=0.001, with_meshcat=False)

    def test_model_directive_serialize_deserialize(self):
        # Can we serialize + deserialize to the model directive format,
        # and recover the same scene (as far as a Drake MBP is concerned)?
        os.environ['ROS_PACKAGE_PATH'] = os.environ['ROS_PACKAGE_PATH'] + ":/tmp/pkg"
        with self.subTest("serialization"):
            # Make sure we can spit out the appropriate files.
            serialize_scene_tree_to_package(
                self.scene_tree,
                package_name='test_save',
                package_parent_dir="/tmp/pkg/",
                remove_directory=True)
            self.assertTrue(os.path.isfile("/tmp/pkg/test_save/package.xml"))
            self.assertTrue(os.path.isfile("/tmp/pkg/test_save/scene_tree.yaml"))
            self.assertTrue(os.path.isdir("/tmp/pkg/test_save/sdf/"))
            # This, at least, should always generate, since a floor always generates.
            self.assertTrue(os.path.isfile("/tmp/pkg/test_save/sdf/floor_0::model_prims.sdf"))
        with self.subTest("deserialization"):
            # Make sure we can load it back in and simulate it.
            mbp_wrangler = PackageToMbpAndSgBuilder(package_dir="/tmp/pkg/test_save")
            mbp_wrangler.Finalize()
        with self.subTest("serialize_deserialize_match"):
            # Double check the set of bodies and frames are identical, and
            # all frames are at the same position.
            _, orig_mbp, _ = compile_scene_tree_to_mbp_and_sg(self.scene_tree)
            orig_mbp.Finalize()
            loaded_mbp = mbp_wrangler.mbp

            orig_context = orig_mbp.CreateDefaultContext()
            loaded_context = loaded_mbp.CreateDefaultContext()
            
            for model_k in range(orig_mbp.num_model_instances()):
                model_id = ModelInstanceIndex(model_k)
                model_name = orig_mbp.GetModelInstanceName(model_id)
                self.assertTrue(loaded_mbp.HasModelInstanceNamed(model_name),
                                "Missing model %s" % model_name)
                corresponding_model_id = loaded_mbp.GetModelInstanceByName(model_name)
                expected_body_ids = orig_mbp.GetBodyIndices(model_id)
                for body_id in expected_body_ids:
                    body = orig_mbp.get_body(body_id)
                    self.assertTrue(loaded_mbp.HasBodyNamed(name=body.name(), model_instance=corresponding_model_id),
                                    "Missing body %s in model %s" % (body.name(), model_name))

            np.testing.assert_allclose(orig_q, loaded_q, 1E-6)
        with self.subTest("deserialize_simulation"):
            # Set up and run some brief sim, and make sure Drake is OK with it.
            diagram = mbp_wrangler.builder.Build()
            diag_context = diagram.CreateDefaultContext()
            sim = Simulator(diagram)
            sim.set_target_realtime_rate(1.0)
            sim.AdvanceTo(0.001)



if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = TestLoader()
    # Run test for 10 different seeds.
    for seed in range(10):
        test = None
        test = FixedSeedKitchenGrammarTests
        test.fixed_seed = seed
        tests = loader.loadTestsFromTestCase(test)
        suite.addTest(tests)
    unittest.TextTestRunner().run(suite)