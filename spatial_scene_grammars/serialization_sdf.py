from copy import deepcopy
import networkx as nx
import numpy as np
import os
import shutil
import yaml
from lxml import etree as et
et.register_namespace('drake', 'drake.mit.edu')

import torch

import pydrake
from pydrake.all import (
    AngleAxis,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    FixedOffsetFrame,
    MultibodyPlant,
    Parser,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
)
import pydrake.geometry as pydrake_geom

from .nodes import SpatialNode, PhysicsGeometryNode, default_friction
from .tree import get_tree_root
from .drake_interop import torch_tf_to_drake_tf, resolve_catkin_package_path


def make_sdf_root():
    # SDF tag
    sdf_root = et.Element("sdf", version="1.6")
    return sdf_root


def add_pose_tag(parent_item, tf, relative_to=None):
    assert isinstance(tf, RigidTransform)
    attrib = {}
    if relative_to:
        attrib["relative_to"] = relative_to
    pose_item = et.SubElement(parent_item, "pose", attrib=attrib)
    xyz = tf.translation()
    rpy = RollPitchYaw(tf.rotation()).vector()
    pose_item.text = "%f %f %f %f %f %f" % (
        xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]
    )
    print("Tf " , tf.GetAsMatrix4(), " to string ", pose_item.text)


def add_inertial_tag(parent_item, mass, com=None, I=None):
    # Inertial info
    inertia = node.spatial_inertia
    inertial_item = et.SubElement(link_item, 'inertial')
    add_pose_tag(inertial_item, RigidTransform(p=inertia.get_com()))
    mass_item = et.SubElement(inertial_item, 'mass')
    mass_item.text = '{:.2E}'.format(inertia.get_mass())
    I = [['{:.4E}'.format(y) for y in x]  # NOQA
         for x in inertia.CalcRotationalInertia().CopyToFullMatrix3()[:]]
    inertia_item = et.SubElement(inertial_item, 'inertia')
    for name, val in zip(["ixx", "ixy", "ixz", "iyy", "iyz", "izz"],
                    [I[0][0], I[0][1], I[0][2], I[1][1], I[1][2], I[2][2]]):
        i_subelem = et.SubElement(inertia_item, name)
        i_subelem.text = str(val)


def add_node_to_sdf_root(sdf_root, node, link_name, ignore_static=False):
    # If ignore_static is True, we'll not indicate that this model
    # is static even if the node is. This can be useful if the
    # model is going to be programmatically welded to another model
    # down the road, in which case this static flag can cause issues
    # (see Drake issue #12227).
    print("Node ", node.name, " with tf ", node.tf)
    if not isinstance(node, PhysicsGeometryNode):
        return

    model_item = et.SubElement(sdf_root, "model", name=node.name)
    link_item = et.SubElement(model_item, "link", name=link_name)
    add_pose_tag(model_item, torch_tf_to_drake_tf(node.tf))

    static_item = et.SubElement(model_item, "static")
    if not ignore_static and node.fixed:
        static_item.text = "1"
    else:
        static_item.text = "0"

    # Inertial info
    inertia = node.spatial_inertia
    inertial_item = et.SubElement(link_item, 'inertial')
    add_pose_tag(inertial_item, RigidTransform(p=inertia.get_com()))
    mass_item = et.SubElement(inertial_item, 'mass')
    mass_item.text = str(inertia.get_mass())
    I = [['{:.4E}'.format(y) for y in x]  # NOQA
         for x in inertia.CalcRotationalInertia().CopyToFullMatrix3()[:]]
    inertia_item = et.SubElement(inertial_item, 'inertia')
    for name, val in zip(["ixx", "ixy", "ixz", "iyy", "iyz", "izz"],
                    [I[0][0], I[0][1], I[0][2], I[1][1], I[1][2], I[2][2]]):
        i_subelem = et.SubElement(inertia_item, name)
        i_subelem.text = str(val)
    
    def make_common_geometry_items(parent_item, tf, geometry):
        # Parent item is "visual" or "collision" item
        add_pose_tag(parent_item, tf)
        geometry_item = et.SubElement(parent_item, "geometry")
        if isinstance(geometry, pydrake_geom.Box):
            box_item = et.SubElement(geometry_item, "box")
            size_item = et.SubElement(box_item, "size")
            size_item.text = "%f %f %f" % (
                geometry.width(),
                geometry.depth(),
                geometry.height()
            )
        elif isinstance(geometry, pydrake_geom.Sphere):
            sphere_item = et.SubElement(geometry_item, "sphere")
            radus_item = et.SubElement(sphere_item, "radus")
            radius.text = "%f" % geometry_item.radius()
        elif isinstance(geometry, pydrake_geom.Cylinder):
            cylinder_item = et.SubElement(geometry_item, "cylinder")
            radius_item = et.SubElement(cylinder_item, "radius")
            radius_item.text = "%f" % geometry.radius()
            length_item = et.SubElement(cylinder_item, "length")
            length_item.text = "%f" % geometry.length()
        elif isinstance(geometry, pydrake_geom.Mesh) or isinstance(geometry, pydrake_geom.Convex):
            mesh_item = et.SubElement(geometry_item, "mesh")
            uri_item = et.SubElement(mesh_item, "uri")
            uri_item.text = geometry.filename()
            scale_item = et.SubElement(mesh_item, "scale")
            scale_item.text = "%f %f %f" % (
                geometry.scale(),
                geometry.scale(),
                geometry.scale()
            )
            if isinstance(geometry, pydrake_geom.Convex):
                et.SubElement(mesh_item, '{drake.mit.edu}declare_convex')
        else:
            raise NotImplementedError("Geometry type ", geometry)

    # Add geometries.
    for k, (tf, geometry, color) in enumerate(node.visual_geometry):
        visual_item = et.SubElement(link_item, "visual", name="visual_%d" % k)
        make_common_geometry_items(visual_item, torch_tf_to_drake_tf(tf), geometry)
        material_item = et.SubElement(visual_item, "material")
        diffuse_item = et.SubElement(material_item, "diffuse")
        diffuse_item.text = "%f %f %f %f" % (color[0], color[1], color[2], color[3])

    for k, (tf, geometry, friction) in enumerate(node.collision_geometry):
        collision_item = et.SubElement(link_item, "collision", name="collision_%d" % k)
        make_common_geometry_items(collision_item, torch_tf_to_drake_tf(tf), geometry)
        geometry_item = et.SubElement(collision_item, "geometry")
        compliance_item = et.SubElement(visual_item, "drake_compliance")
        static_friction = et.SubElement(compliance_item, "static_friction")
        static_friction.text = "%f" % friction.static_friction()
        dynamic_friction = et.SubElement(compliance_item, "dynamic_friction")
        dynamic_friction.text = "%f" % friction.dynamic_friction()

    return


def save_sdf_with_node_geometry(node, path, link_name, ignore_static=False):
    ''' Given a node, creates an SDF that has just this node's
    geometry in it. Returns True if an SDF was created, or False if not.
    (If the node doesn't have geometry, no SDF will be made.) '''
    if len(node.collision_geometry) == 0 and len(node.visual_geometry) == 0:
        return False

    sdf_root = make_sdf_root()
    add_node_to_sdf_root(sdf_root, node, link_name, ignore_static=ignore_static)
    
    need_dir, _ = os.path.split(path)
    os.makedirs(need_dir, exist_ok=True)
    et.ElementTree(sdf_root).write(path, pretty_print=True)
    return True


def add_subtree_to_sdf_root(sdf_root, scene_tree, node):
    # Recursively add each node to the tree, adding welds and
    # relative poses appropriately.
    add_node_to_sdf_root(sdf_root, node, node.name)
    for rule in scene_tree.successors(node):
        for child_node in scene_tree.successors(rule):
            # TODO: We could have the appropriate hierarchy of welds
            # by looking at parent/child relationships here.
            # But since things are either fixed to the world or floating,
            # it doesn't make a difference...
            add_subtree_to_sdf_root(sdf_root, scene_tree, child_node)


def serialize_scene_tree_to_package_and_single_sdf(
        scene_tree, out_sdf_name):
    ''' Given a scene tree, serializes it out to one large SDF file that
    references all of the relevant models. '''

    sdf_root = make_sdf_root()
    root_node = get_tree_root(scene_tree)
    add_subtree_to_sdf_root(sdf_root, scene_tree, root_node)

    et.ElementTree(sdf_root).write(out_sdf_name, pretty_print=True)



