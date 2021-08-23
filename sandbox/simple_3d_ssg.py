'''
Minimal simple grammar that hopefully enables MIP-based MAP
scene parsing.

Spatial scene grammar definition: each node has:
- a pose `tf` \in SE(3), represented with a RigidTransform
- a shape, as a Meshcat-visualizable geometry.

Nodes can choose which child to produce by following:
- an AND rule, producing all possible children.
- an OR rule, producing one child out of the set of possible children.
- a GEOMETRIC_SET rule, producing up to N children.

For each child, if it is produced, nodes can produce the child:
- With xyz:
   - In a uniform xyz window offset relative to parent
- With rotation:
   - Unconstrained
   - In uniform angle window, rotating around specified axis in parent frame.

A node subclasses a Node type. They should be initializable with zero
arguments.
'''

import networkx as nx
import numpy as np
import random

from copy import deepcopy
from collections import namedtuple

import pydrake
from pydrake.all import (
    AngleAxis,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    UniformlyRandomRotationMatrix
)

import meshcat
import meshcat.geometry as meshcat_geom
import meshcat.transformations as meshcat_tf

# Axis-aligned, for now. Can generalize to convex set.
XyzBounds = namedtuple("xyz_bounds",
    ["xyz_min", "xyz_max"]
)
# TODO: Should be able to specify the rotation offset
# in the child frame. It's currently assumed that at
# 0 rotation around the axis, the child frame lines
# up with the parent frame; and that from the child
# frame's reference, the rotation happens around the +z axis.
RotationBounds = namedtuple("rotation_bounds",
    ["axis", "min_angle", "max_angle"]
)
# child_constructor takes the child tf, and produces the child
# node 
ChildInfo = namedtuple("child_info",
    ["child_type", "child_xyz_bounds", "child_rotation_bounds"]
)

class Node(object):
    '''
        tf: RigidTransform pose of node
        observed: Boolean, whether this node is observable
        geometry: Meshcat geometry to visualize node.
    '''
    def __init__(self, tf, observed, geometry,
                 geometry_scale=np.ones(3), geometry_tf=RigidTransform(),
                 geometry_color=0xffaa55):
        self._tf = tf
        self._observed = observed
        self._geometry = geometry
        self._geometry_scale = geometry_scale
        self._geometry_tf = geometry_tf
        self._geometry_color = geometry_color
        super().__init__()

    @property
    def tf(self):
        return self._tf
    
    @tf.setter
    def tf(self, tf):
        self._tf = tf
        
    @property
    def observed(self):
        return self._observed
    
    @property
    def geometry(self):
        return self._geometry
    
    @property
    def geometry_scale(self):
        return self._geometry_scale

    @property
    def geometry_tf(self):
        return self._geometry_tf
    
    @property
    def geometry_color(self):
        return self._geometry_color
    
    def sample_children(self):
        raise NotImplementedError("Implement sample_children in subclass.")
    
    @staticmethod
    def _sample_child(parent_tf, child_info):
        # Given a ChildInfo struct, produce a randomly sampled child.
        # Sample child xyz and rotation
        xyz_offset = np.random.uniform(
            low=child_info.child_xyz_bounds.xyz_min,
            high=child_info.child_xyz_bounds.xyz_max
        )
        if child_info.child_rotation_bounds is None:
            # No constraints on child rotation; randomly
            # sample a rotation instead.
            R_offset = UniformlyRandomRotationMatrix(RandomGenerator(np.random.randint(2**31)))
        else:
            axis = child_info.child_rotation_bounds.axis
            angle = np.random.uniform(
                low=child_info.child_rotation_bounds.min_angle,
                high=child_info.child_rotation_bounds.max_angle
            )
            R_offset = RotationMatrix(AngleAxis(angle=angle, axis=axis))
        # Child XYZ is in *world* frame, with with translational offset.
        #tf_offset = RigidTransform(p=parent_tf.rotation().inverse().multiply(xyz_offset), R=R_offset)
        #tf = parent_tf.multiply(tf_offset)
        tf = RigidTransform(
            p = parent_tf.translation() + xyz_offset,
            R = parent_tf.rotation().multiply(R_offset)
        )
        return child_info.child_type(tf=tf)


class TerminalNode(Node):
    ''' Node type that produces no children. '''
    def sample_children(self):
        return []

    
class AndNode(Node):
    ''' Node type that produces all children. '''
    def __init__(self, child_infos, **kwargs):
        assert len(child_infos) > 0
        assert all([isinstance(c, ChildInfo) for c in child_infos])
        self.child_infos = child_infos
        super().__init__(**kwargs)
    
    def sample_children(self):
        children = []
        for child_info in self.child_infos:
            children.append(self._sample_child(self.tf, child_info))
        return children

class OrNode(Node):
    ''' Node type that produces all children. '''
    def __init__(self, child_infos, child_probs, **kwargs):
        assert len(child_infos) > 0
        assert all([isinstance(c, ChildInfo) for c in child_infos])
        assert len(child_infos) == len(child_probs)
        self.child_infos = child_infos
        self.child_probs = child_probs
        super().__init__(**kwargs)
    
    def sample_children(self):
        # Pick which child will be produced.
        child_ind = np.random.choice(len(self.child_infos), p=self.child_probs)
        children = [self._sample_child(self.tf, self.child_infos[child_ind])]
        return children
        
    
class GeometricSetNode(Node):
    ''' Node type that produces all children. '''
    def __init__(self, child_infos, p, max_children, **kwargs):
        assert len(child_infos) == 1
        assert all([isinstance(c, ChildInfo) for c in child_infos])
        self.child_infos = child_infos
        self.p  = p
        self.max_children = max_children
        super().__init__(**kwargs)
    
    def sample_children(self):
        children = []
        # Repeatedly spawn
        n = min(np.random.geometric(self.p), self.max_children)
        for k in range(n):
            children.append(self._sample_child(self.tf, self.child_infos[0]))
        return children
    
def sample_tree(root_type, root_tf):
    # Sample an unconditioned tree from this grammar.
    tree = nx.DiGraph()
    root = root_type(tf=root_tf)
    tree.add_node(root)
    node_queue = [root]

    while len(node_queue) > 0:
        parent = node_queue.pop(0)
        # Ask node to sample its children.
        children = parent.sample_children()
        for child in children:
            tree.add_node(child)
            tree.add_edge(parent, child)
            node_queue.append(child)

    return tree

def get_observed_nodes(tree):
    # Given a scene tree (nx.DiGraph) and a list of observed
    # node type names (list of strings), pulls out only nodes
    # in the  tree with matching type into a list of Nodes.
    return [n for n in tree if n.observed is True]

def get_tree_root(tree):
    # Assumes tree structure; might infinite loop otherwise.
    root = list(tree.nodes)[0]
    while len(list(tree.predecessors(root))) > 0:
        root = list(tree.predecessors(root))[0]
    return root
        
def draw_tree(tree, vis, prefix="", draw_regions=False):
    # Given a scene tree (nx.DiGraph), draw it in the
    # specified meshcat visualizer.
    
    # Draw the scene geometry flat, to keep TFs easy.
    name_prefix = prefix + "scene"
    vis[name_prefix].delete()
    k = 0
    for node in tree.nodes:
        name = name_prefix + "/%s_%03d" % (node.__class__.__name__, k)
        if node.geometry is not None:
            color = node.geometry_color
            alpha = 1.0
            vis[name].set_object(
                node.geometry,
                meshcat_geom.MeshLambertMaterial(color=color, opacity=alpha, transparent=(alpha != 1.))
            )
            tf = node.tf.GetAsMatrix4()
            geom_tf = node.geometry_tf.GetAsMatrix4()
            tf = tf.dot(geom_tf)
            tf[:3, :3] = tf[:3, :3].dot(np.diag(node.geometry_scale))
            print(tf)
            vis[name].set_transform(tf)
            k += 1
    
    # Draw the tree structure.
    tree_prefix = prefix + "tree"
    vis[tree_prefix].delete()
    k = 0
    for node in tree.nodes:
        name = tree_prefix + "/" + node.__class__.__name__ + "_%03d" % k
        k += 1
        # Draw node as randomly colored sphere
        color = random.randint(0, 0xFFFFFF)
        alpha = 0.5
        vis[name]["triad"].set_object(
            meshcat_geom.triad(scale=0.1)
        )
        vis[name]["sphere"].set_object(
            meshcat_geom.Sphere(0.01),
            meshcat_geom.MeshToonMaterial(color=color, opacity=alpha, transparent=(alpha != 1.))
        )
        vis[name].set_transform(node.tf.GetAsMatrix4())
        # Draw children
        verts = []
        for child in tree.successors(node):
            # Draw link to child
            verts.append(node.tf.translation()),
            verts.append(child.tf.translation())
        if len(verts) > 0:
            verts = np.vstack(verts).T
            # Don't want this as a direct child or it'll inherit the transform
            vis[name + "_child_connections"].set_object(
                meshcat_geom.Line(meshcat_geom.PointsGeometry(verts),
                                  meshcat_geom.LineBasicMaterial(linewidth=50, color=color)))
        
        if draw_regions:
            # Draw the child regions for each child
            if isinstance(node, (AndNode, OrNode, GeometricSetNode)):
                for info_k, child_info in enumerate(node.child_infos):
                    region_name = "child_region_%03d" % info_k
                    lb = child_info.child_xyz_bounds.xyz_min
                    ub = child_info.child_xyz_bounds.xyz_max
                    vis[name][region_name].set_object(
                        meshcat_geom.Box(ub - lb),
                        meshcat_geom.MeshToonMaterial(color=0x111111, opacity=0.1, transparent=True)
                    )
                    tf = RigidTransform(p=(ub+lb)/2)
                    vis[name][region_name].set_transform(tf.GetAsMatrix4())