import pyro
import pyro.distributions as dist
from pyro import poutine

from scene_grammar.src.nodes import *
from scene_grammar.src.rules import *
from scene_grammar.src.tree import *
from scene_grammar.src.transform_utils import *
from scene_grammar.src.visualization import *
from scene_grammar.src.drake_interop import *
from scene_grammar.src.serialization import *

from grammar_room_layout import *

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.manual_seed(42)
    pyro.clear_param_store()
    trace = pyro.poutine.trace(
        ParseTree.generate_from_root_type).get_trace(
        root_node_type=Kitchen,
        name="kitchen",
        tf=torch.eye(4))
    print(trace.format_shapes())
