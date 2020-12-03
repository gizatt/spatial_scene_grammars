import meshcat
import meshcat.geometry as meshcat_geom
import matplotlib.pyplot as plt
import numpy as np

from spatial_scene_grammars.transform_utils import *
from spatial_scene_grammars.tree import get_tree_root

def rgb_2_hex(rgb):
    # Turn a list of R,G,B elements (any indexable list
    # of >= 3 elements will work), where each element is
    # specified on range [0., 1.], into the equivalent
    # 24-bit value 0xRRGGBB.
    val = 0
    for i in range(3):
        val += (256**(2 - i)) * int(255 * rgb[i])
    return val


def draw_scene_tree_meshcat(scene_tree, zmq_url=None, alpha=1.0, node_sphere_size=0.1):
    pruned_tree = scene_tree.get_tree_without_production_rules()

    # Do actual drawing in meshcat, starting from root of tree
    # So first find the root...
    root_node = get_tree_root(pruned_tree)

    vis = meshcat.Visualizer(zmq_url=zmq_url or "tcp://127.0.0.1:6000")
    vis["scene_tree"].delete()
    node_queue = [root_node]
    
    # Assign functionally random colors to each new node
    # type we discover.
    node_class_to_color_dict = {}
    cmap = plt.cm.get_cmap('jet')
    cmap_counter = 0.
    
    k = 0
    while len(node_queue) > 0:
        node = node_queue.pop(0)
        children = list(pruned_tree.successors(node))
        node_queue += children
        # Draw this node
        node_type_string = node.__class__.__name__
        if node_type_string in node_class_to_color_dict.keys():
            color = node_class_to_color_dict[node_type_string]
        else:
            color = rgb_2_hex(cmap(cmap_counter))
            node_class_to_color_dict[node_type_string] = color
            cmap_counter = np.fmod(cmap_counter + np.pi*2., 1.)

        vis["scene_tree"][node.name + "%d" % k].set_object(
            meshcat_geom.Sphere(node_sphere_size),
            meshcat_geom.MeshToonMaterial(color=color, opacity=alpha, transparent=(alpha != 1.)))

        tf = node.tf.cpu().detach().numpy()
        vis["scene_tree"][node.name + "%d" % k].set_transform(tf)

        # Draw connections to children
        verts = []
        for child in children:
            verts.append(node.tf[:3, 3].cpu().detach().numpy())
            verts.append(child.tf[:3, 3].cpu().detach().numpy())
        if len(verts) > 0:
            verts = np.vstack(verts).T
            vis["scene_tree"][node.name + "%d" % k + "_child_connections"].set_object(
                meshcat_geom.Line(meshcat_geom.PointsGeometry(verts),
                                  meshcat_geom.LineBasicMaterial(linewidth=10, color=color)))
        k += 1