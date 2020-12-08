from collections import namedtuple
from copy import deepcopy
import networkx as nx
import numpy as np
import os
import shutil
import yaml
from lxml import etree as et

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

# Helper for pose serialization into YAML
class RpyTag(yaml.YAMLObject):
    yaml_tag = u'!Rpy'

    def __init__(self, deg):
        self.deg = deg
    @classmethod
    def from_yaml(cls, loader, node):
        info = loader.construct_mapping(node, deep=True)
        return RpyTag(deg=info["deg"])


def make_dict_from_tf(tf=None):
    if tf is None:
        return {}
    assert isinstance(tf, RigidTransform)        
    translation = tf.translation().tolist()
    rpy_deg = RollPitchYaw(tf.rotation()).vector()*180./np.pi
    rotation = RpyTag(rpy_deg.tolist())
    return {
        "translation": translation,
        "rotation": rotation
    }

def make_tf_from_dict(x_pf_dict):
    trans = x_pf_dict["translation"]
    rpy = RollPitchYaw(np.array(x_pf_dict["rotation"].deg)*np.pi/180.)
    return x_pf_dict["base_frame"], RigidTransform(p=trans, rpy=rpy)

def add_model_directive(model_name, model_path):
    return {"add_model":
        { "name": model_name,
          "file": model_path
        }
    }


def add_frame_directive(frame_name, base_frame_name, tf=None):
    x_pf_base = make_dict_from_tf(tf)
    x_pf_base["base_frame"] = base_frame_name
    return {
        "add_frame":
        { 
            "name": frame_name,
            "X_PF": x_pf_base
        }
    }


def add_weld_directive(parent_frame_name, child_frame_name):
    weld_info_base = {}
    weld_info_base["parent"] = parent_frame_name
    weld_info_base["child"] = child_frame_name
    return {
        "add_weld": weld_info_base
    }


def set_initial_free_body_pose_directive(model_name, body_name, base_frame_name, tf):
    rel_pose_info_base = make_dict_from_tf(tf)
    rel_pose_info_base["base_frame"] = base_frame_name
    return {
        "set_initial_free_body_pose": {
            "X_PF": rel_pose_info_base,
            "body_name": body_name,
            "model_name": model_name
        }
    }

def set_initial_configuration_directive(model_name, q0):
    assert isinstance(q0, dict)
    for key in list(q0.keys()):
        if isinstance(key, np.ndarray):
            q0[key] = q0[key].tolist()
        else:
            q0[key] = float(q0[key])
    return {
        "set_initial_configuration": {
            "model_name": model_name,
            "q0": q0
        }
    }


def save_sdf_with_node_geometry(node, path, root_link_name):
    ''' Returns True if an SDF was created, or False if not. '''
    if len(node.collision_geometry) == 0 and len(node.visual_geometry) == 0:
        return False

    # Build up SDF from the bottom up.
    # Start with registering Drake namespace
    et.register_namespace('drake', 'drake.mit.edu')
    # SDF tag
    sdf_root = et.Element("sdf", version="1.6")
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

    need_dir, _ = os.path.split(path)
    os.makedirs(need_dir, exist_ok=True)
    et.ElementTree(sdf_root).write(path, pretty_print=True)
    return True

def build_directives_for_node_geometry(node, base_frame_name, package_name, package_parent_dir):
    assert isinstance(node, PhysicsGeometryNode)
    directives = []
    # For disambiguating naming
    model_k = 0

    # Collapse everything into adding-models-from-SDF
    # format.
    model_info_to_add = deepcopy(node.model_paths)
    
    # Spit out an SDF with all of the visual / geometry info, if there's any
    primitive_tf = torch.eye(4)
    within_package_model_path = os.path.join("sdf", "%s::model_prims.sdf" % node.name)
    # Replace :: with __ to make a safer filename
    # within_package_model_path = within_package_model_path.replace("::", "__")
    primitive_root_body = node.name
    if (save_sdf_with_node_geometry(
            node, os.path.join(package_parent_dir, package_name, within_package_model_path),
            primitive_root_body)):
        primitive_model_path_with_pkg = "%s://%s" % (package_name, within_package_model_path)
        model_info_to_add.append((
            primitive_tf, primitive_model_path_with_pkg, primitive_root_body, None
        ))

    for (tf, model_path, root_body_name, q0_dict) in model_info_to_add:
        # Add the model itself
        model_name = "%s::model_%d" % (node.name, model_k)
        model_k += 1
        full_body_name = "%s::%s" % (model_name, root_body_name)
        
        directives.append(add_model_directive(
            model_name=model_name,
            model_path=model_path
        ))
        if node.fixed:
            # Weld the root body of the model to the pose specified.
            model_name = model_name + "_frame"
            directives.append(add_frame_directive(
                frame_name=model_name,
                base_frame_name=base_frame_name,
                tf=torch_tf_to_drake_tf(tf)
            ))
            directives.append(add_weld_directive(
                parent_frame_name=model_name,
                child_frame_name=full_body_name
            ))
        else:
            # Just supply an initial pose for the body.
            directives.append(set_initial_free_body_pose_directive(
                model_name=model_name,
                body_name=root_body_name,
                base_frame_name=base_frame_name,
                tf=torch_tf_to_drake_tf(tf)
            ))
        # If there's an initial configuration, supply it
        if q0_dict is not None:
            directives.append(set_initial_configuration_directive(
                model_name=model_name,
                q0=q0_dict
            ))

    # TODO: Clearance geometry is now discarded. Is that OK?
    return directives

def build_directives_for_node(
    scene_tree, node, base_frame_name, tf,
    prefix, package_name, package_parent_dir):
    # Given a node, will build directives and write files to:
    # - Add a base frame for self, relative to parent, if existing
    # - Add models for primitive + existing-model geometry
    # - Add offset frames and weld commands for fixed geometry
    # - Add initial pose commands for non-fixed geometry
    directives = []
    my_frame_name = None
    if isinstance(node, SpatialNode):
        my_frame_name = node.name + "_frame"
        # Create our own root frame.
        directives.append(add_frame_directive(
            my_frame_name, base_frame_name, tf
        ))
        if isinstance(node, PhysicsGeometryNode):
            directives += build_directives_for_node_geometry(node, my_frame_name, 
                package_name=package_name,
                package_parent_dir=package_parent_dir)
    
    # For each child, call recursively.
    for rule in scene_tree.successors(node):
        for child_node in scene_tree.successors(rule):
            child_rel_tf = None
            if isinstance(child_node, SpatialNode) and isinstance(node, SpatialNode):
                my_tf = torch_tf_to_drake_tf(node.tf)
                child_tf = torch_tf_to_drake_tf(child_node.tf)
                child_rel_tf = my_tf.inverse().multiply(child_tf)
            directives += build_directives_for_node(
                scene_tree, child_node,
                base_frame_name=my_frame_name or "WorldBody",
                tf=child_rel_tf,
                prefix=prefix,
                package_name=package_name,
                package_parent_dir=package_parent_dir
            )
    return directives

def make_default_package_xml(package_name, path):
    template_str = """<package format="2">
<name>%s</name>
<version>0.0.0</version>
<description>
    Auto-generated saved scene tree.
</description>
<maintainer email="fillme">No One</maintainer>
<license>MIT</license>
</package>"""
    need_dir, _ = os.path.split(path)
    os.makedirs(need_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(template_str % package_name)

def serialize_scene_tree_to_package(
        scene_tree, package_name="saved_scene_tree",
        package_parent_dir="./out/",
        remove_directory=False):
    ''' Given a scene tree, serializes it out to a top-level YAML that
    lists out SDF/URDF models, specifies the transform (and indicates
    which link is the root link, if it's a complex model file), and
    indicates whether it's welded to the world.

    This is close to Drake model directives format, but is modified
    to support specifying initial poses of floating models and specify
    relative transforms for welds.

    This will auto-generate SDFs containing the primitive geometry in the
    scene. '''

    if remove_directory:
        shutil.rmtree(os.path.join(package_parent_dir, package_name), ignore_errors=True)

    directives = build_directives_for_node(
        scene_tree, get_tree_root(scene_tree),
        tf=RigidTransform(), prefix=None, base_frame_name="WorldBody",
        package_name=package_name, package_parent_dir=package_parent_dir)

    # Save that out as the model directions YAML
    os.makedirs(os.path.join(package_parent_dir, package_name), exist_ok=True)
    with open(os.path.join(package_parent_dir, package_name, "scene_tree.yaml"), "w") as f:
        yaml.dump(directives, f, default_flow_style=False)

    # Create a simple package.xml so the generated SDFs
    # can be referred to by "<package_name>://<internal path>"
    make_default_package_xml(
        package_name=package_name,
        path=os.path.join(package_parent_dir, package_name, "package.xml")
    )


def get_frame_from_full_name(mbp, full_name):
    # Figures out of a frame name like XX:YY:ZZ
    # is referring to a frame by name "XX:YY:ZZ"
    # or a model "XX:YY" with frame "ZZ"
    try:
        return mbp.GetFrameByName(full_name)
    except RuntimeError:
        pass
    parts = full_name.split("::")
    model_name = parts[:-1]
    frame_name = parts[-1]
    # If there's no model name, let Drake throw the
    # error that this frame doesn't exist.
    if len(model_name) == 0:
        return mbp.GetFrameByName(frame_name)
    # Otherwise get the indicated model and subframe.
    model_id = mbp.GetModelInstanceByName("::".join(model_name))
    return mbp.GetFrameByName(model_instance=model_id, name=frame_name)


class PackageToMbpAndSgBuilder():
    ''' Builds an MBP from a scene tree serialized as a catkin
    package and model directive YAML. '''
    def __init__(self, package_dir, timestep=0.001):
        '''
        Args:
            - package_dir: Directory that contains at least `scene_tree.yaml`
            and `package.xml` files.
            - timestep: Simulation timestep for Drake MBP.
        
        Will initialize an object of this class with mbp, scene_graph, and
        builder (DiagramBuilder) members to be used by someone else
        to build up a diagram for sim.
        '''
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        scene_tree_yaml_path = os.path.join(package_dir, "scene_tree.yaml")
        assert os.path.isfile(scene_tree_yaml_path)

        # Set up the MBP we'll be building up.
        builder = DiagramBuilder()
        mbp, scene_graph = AddMultibodyPlantSceneGraph(
            builder, MultibodyPlant(time_step=timestep))
        parser = Parser(mbp)
        # Make sure Parser looks for everything on ROS_PACKAGE_PATH
        print(os.environ['ROS_PACKAGE_PATH'])
        parser.package_map().PopulateFromEnvironment("ROS_PACKAGE_PATH")
        world_body = mbp.world_body()

        # Keep track of body poses we need to update
        self.free_body_poses = []

        # Register AngleAxis type with the YAML loader.
        yaml.SafeLoader.add_constructor(
            RpyTag.yaml_tag,
            RpyTag.from_yaml)
        with open(scene_tree_yaml_path, "r") as f:
            scene_tree_yaml = yaml.safe_load(f)
        
        # Each directive maps directly to an MBP command, so step through and apply them.
        for directive in scene_tree_yaml:
            keys = list(directive.keys())
            assert len(keys) == 1, "Malformed directives file: directive has 0 or > 1 keys."
            key = keys[0]
            vals = directive[key]

            if key == "add_frame":
                # TODO: What model instance should I add each frame to? Currently
                # they all wind up in world, I think... which is fine, because
                # they all have unique full names.
                base_frame_name, tf = make_tf_from_dict(vals["X_PF"])
                parent_frame = get_frame_from_full_name(mbp, base_frame_name)
                new_frame = FixedOffsetFrame(name=vals["name"], P=parent_frame, X_PF=tf)
                mbp.AddFrame(new_frame)
            elif key == "add_weld":
                mbp.WeldFrames(
                    get_frame_from_full_name(mbp, vals["parent"]),
                    get_frame_from_full_name(mbp, vals["child"])
                )
            elif key == "add_model":
                parser.AddModelFromFile(
                    resolve_catkin_package_path(parser.package_map(), vals["file"]),
                    vals["name"]
                )
            elif key == "set_initial_configuration":
                q0_dict = vals['q0_dict']
                model_id = mbp.GetModelInstanceByName(vals["model_name"])
                for joint_name in list(q0_dict.keys()):
                    q0_this = q0_dict[joint_name]
                    joint = mbp.GetMutableJointByName(
                        joint_name, model_instance=model_id)
                    # Reshape to make Drake happy.
                    q0_this = q0_this.reshape(joint.num_positions(), 1)
                    joint.set_default_positions(q0_this)
            elif key == "set_initial_free_body_pose":
                model_id = mbp.GetModelInstanceByName(vals["model_name"])
                if "body_name" in vals and vals["body_name"] is not None:
                    body_name = vals["body_name"]
                    body = mbp.GetBodyByName(name=body_name, model_instance=model_id)
                # Try to guess body name if it wasn't provided.
                else:
                    body_ind_possibilities = mbp.GetBodyIndices(model_id)
                    assert len(body_ind_possibilities) == 1, \
                        "Please supply root_body_name for model with path %s" % model_path
                    body = mbp.get_body(body_ind_possibilities[0])
                # Unfortunately SetDefaultFreeBodyPose does not support frames other
                # than the world frame; we'll have to set these up later once we know
                # the full tree structure.
                base_frame_name, offset_tf = make_tf_from_dict(vals["X_PF"])
                self.free_body_poses.append((offset_tf, base_frame_name, body))

            else:
                raise NotImplementedError("Unsupported directive %s" % key)
        
            self.mbp = mbp
            self.builder = builder
            self.scene_graph = scene_graph

    def Finalize(self):
        # Because of the weirdness with MBP, we can't set free body
        # poses until we finalize the MBP. When that's ready to be
        # done (all other additions / modifications to the MBP are done),
        # this method handles finalization and setting the default free
        # body poses correctly.
        self.mbp.Finalize()
        context = self.mbp.CreateDefaultContext()
        for (offset_tf, base_frame_name, body) in self.free_body_poses:
            base_tf = self.mbp.CalcRelativeTransform(
                context, self.mbp.world_frame(),
                get_frame_from_full_name(self.mbp, base_frame_name))
            print(base_tf.GetAsMatrix4())
            print(offset_tf.GetAsMatrix4())
            self.mbp.SetDefaultFreeBodyPose(body, base_tf.multiply(offset_tf))
