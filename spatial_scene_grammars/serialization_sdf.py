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


def add_node_as_model_to_sdf_root(sdf_root, node, root_link_name):
    # Overarching model
    model_item = et.SubElement(sdf_root, "model", name=node.name)
    link_item = et.SubElement(model_item, "link", name=root_link_name)
    # Inertial info
    inertia = node.spatial_inertia
    inertia_item = et.SubElement(link_item, 'inertial')
    pose_item = et.SubElement(inertia_item, 'pose')
    pose_item.text = "{:.2E} {:.2E} {:.2E} 0. 0. 0.".format(
                        *inertia.get_com().tolist())
    mass_item = et.SubElement(inertia_item, 'mass')
    mass_item.text = '{:.2E}'.format(inertia.get_mass())
    I = [['{:.4E}'.format(y) for y in x]  # NOQA
         for x in inertia.CalcRotationalInertia().CopyToFullMatrix3()[:]]
    et.SubElement(
        inertia_item,
        'inertia',
        ixx=I[0][0],
        ixy=I[0][1],
        ixz=I[0][2],
        iyy=I[1][1],
        iyz=I[1][2],
        izz=I[2][2])

    def make_common_geometry_items(parent_item, tf, geometry):
        # Parent item is "visual" or "collision" item
        pose_item = et.SubElement(parent_item, "pose")
        x, y, z = tf.translation()
        r, p, y = RollPitchYaw(tf.rotation()).vector()
        pose_item.text = "%f %f %f %f %f %f" % (
            x, y, z, r, p, y
        )
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


def save_sdf_with_node_geometry(node, path, root_link_name):
    ''' Given a node, creates an SDF that has just this node's
    geometry in it. Returns True if an SDF was created, or False if not.
    (If the node doesn't have geometry, no SDF will be made.) '''
    if len(node.collision_geometry) == 0 and len(node.visual_geometry) == 0:
        return False

    sdf_root = make_sdf_root()
    add_node_as_model_to_sdf_root(sdf_root, node, root_link_name)
    
    need_dir, _ = os.path.split(path)
    os.makedirs(need_dir, exist_ok=True)
    et.ElementTree(sdf_root).write(path, pretty_print=True)
    return True
