import pyro
import pyro.distributions as dist
from pyro import poutine

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.tree import *
from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.serialization import *

from spatial_scene_grammars_examples.kitchen.grammar_room_layout import *

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(42)
    pyro.clear_param_store()
    trace = pyro.poutine.trace(
        SceneTree.forward_sample_from_root_type).get_trace(
        root_node_type=Kitchen,
        name="kitchen",
        tf=torch.eye(4))
    print(trace.format_shapes())
