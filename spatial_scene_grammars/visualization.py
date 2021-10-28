import os, contextlib
import meshcat
import meshcat.geometry as meshcat_geom
import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    ConnectMeshcatVisualizer,
)
from .drake_interop import *

def rgb_2_hex(rgb):
    # Turn a list of R,G,B elements (any indexable list
    # of >= 3 elements will work), where each element is
    # specified on range [0., 1.], into the equivalent
    # 24-bit value 0xRRGGBB.
    val = 0
    for i in range(3):
        val += (256**(2 - i)) * int(255 * rgb[i])
    return val

def draw_scene_tree_contents_meshcat(scene_tree, prefix="scene", zmq_url=None, alpha=0.25, draw_clearance_geom=False, quiet=True):
    ''' Given a scene tree, draws it in meshcat at the requested ZMQ url.
        Can be configured to draw the tree geometry or the clearance geometry. '''
    
    if draw_clearance_geom:
        builder, mbp, scene_graph = compile_scene_tree_clearance_geometry_to_mbp_and_sg(scene_tree)
    else:
        builder, mbp, scene_graph, _, _, = compile_scene_tree_to_mbp_and_sg(scene_tree)
    mbp.Finalize()

    if quiet:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                vis = ConnectMeshcatVisualizer(builder, scene_graph,
                    zmq_url=zmq_url or "default", prefix=prefix)
    else:
        vis = ConnectMeshcatVisualizer(builder, scene_graph,
            zmq_url=zmq_url or "default", prefix=prefix)
    vis.delete_prefix()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    vis.load(vis.GetMyContextFromRoot(context))
    diagram.Publish(context)
    # Necessary to manually remove this meshcat visualizer now that we're
    # done with it, as a lot of Drake systems (that are involved with the
    # diagram builder) don't get properly garbage collected. See Drake issue #14387.
    # Meshcat collects sockets, so deleting this avoids a file descriptor
    # leak.
    del vis.vis

def draw_scene_tree_structure_meshcat(scene_tree, prefix="scene_tree", zmq_url=None,
        alpha=0.775, node_sphere_size=0.05, linewidth=2, with_triad=True, quiet=True,
        color_by_score=None):
    # Color by score can be a tuple of min, max score. It'll go from red at min score
    # to blue at max score.
    # Do actual drawing in meshcat, starting from root of tree
    # So first find the root...
    root_node = scene_tree.get_root()

    if quiet:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                vis = meshcat.Visualizer(zmq_url=zmq_url or "tcp://127.0.0.1:6000")
    else:
        vis = meshcat.Visualizer(zmq_url=zmq_url or "tcp://127.0.0.1:6000")

    vis[prefix].delete()
    node_queue = [root_node]
    
    # Assign functionally random colors to each new node
    # type we discover, or color my their scores.
    node_class_to_color_dict = {}
    cmap = plt.cm.get_cmap('jet')
    cmap_counter = 0.

    
    k = 0
    while len(node_queue) > 0:
        node = node_queue.pop(0)
        children, rules = scene_tree.get_children_and_rules(node)
        node_queue += children

        # 
        if color_by_score is not None:
            assert len(color_by_score) == 2, "Color by score should be a tuple of (min, max)"
            score = node.score_child_set(children)
            score = (score - color_by_score[0]) / (color_by_score[1] - color_by_score[0])
            score = 1. - np.clip(score.item(), 0., 1.)
            color = rgb_2_hex(cmap(score))
            color = 0x555555
        else: 
            # Draw this node
            node_type_string = node.__class__.__name__
            if node_type_string in node_class_to_color_dict.keys():
                color = node_class_to_color_dict[node_type_string]
            else:
                color = rgb_2_hex(cmap(cmap_counter))
                node_class_to_color_dict[node_type_string] = color
                cmap_counter = np.fmod(cmap_counter + np.pi*2., 1.)

        vis[prefix][node.name + "%d/sphere" % k].set_object(
            meshcat_geom.Sphere(node_sphere_size),
            meshcat_geom.MeshToonMaterial(color=color, opacity=alpha, transparent=(alpha != 1.), depthTest=False))
        if with_triad:
            vis[prefix][node.name + "%d/triad" % k].set_object(
                meshcat_geom.triad(scale=node_sphere_size*5.)
            )

        tf = node.tf.cpu().detach().numpy()
        vis[prefix][node.name + "%d" % k].set_transform(tf)

        # Draw connections to each child
        for child, rule in zip(children, rules):
            verts = []
            verts.append(node.tf[:3, 3].cpu().detach().numpy())
            verts.append(child.tf[:3, 3].cpu().detach().numpy())
            verts = np.vstack(verts).T

            if color_by_score is not None:
                score = rule.score_child(node, child)
                score = (score - color_by_score[0]) / (color_by_score[1] - color_by_score[0])
                score = 1. - np.clip(score.item(), 0., 1.)
                color = rgb_2_hex(cmap(score))

            vis[prefix][node.name + "_to_" + child.name].set_object(
                meshcat_geom.Line(meshcat_geom.PointsGeometry(verts),
                                  meshcat_geom.LineBasicMaterial(linewidth=linewidth, color=color, depthTest=False)))
            k += 1