import numpy as np
import trimesh
from trimesh.constants import log
from copy import deepcopy
import glob

import os

import open3d
import numpy as np
import cv2
import imutils

from lxml import etree as ET

from pydrake.all import (
    SpatialInertia,
    RotationalInertia
)

def do_visual_mesh_simplification(input_obj_path, target_tris=1000):
    '''
    Given an obj path, simplifies the geometry and texture so
    it renders more easily in meshcat / drake vis by:
    - Creates a (possibly simplified) mesh alongside it, with a 
    "_simple_vis.obj" post-fox.
    - Looks for a texture file at a fixed relative path
    (../material/textures/texture.png), resizes it to max edge length
    1024 to decrease size, and saves it alongside the new saved visual
    mesh with a .png extension.

    No existing files changed -- just new ones added.

    Args:
    - input_obj_path: String path to obj file
    - target_tris: Currently unusued, but would be target for mesh decimation.

    Returns:
    - output obj file path
    '''
    output_obj_path = input_obj_path[:-4] + "_simple_vis.obj"
    output_texture_path = output_obj_path[:-4] + ".png"
    print("Output path: ", output_texture_path)

    mesh = open3d.io.read_triangle_mesh(input_obj_path)
    simplified_mesh = mesh # mesh.simplify_quadric_decimation(target_tris)
    # Can't quite get this to work as Open3D stores UVs per
    # triangle rather than per vertex.
    # I should get the triangle centers and pull the UV from
    # the corresponding triangle rather than doing it by
    # raw vertex.
    # But this whole pipeline is moot if we have a fast enough
    # renderer, so whatever.
    ## Update UVs on simplified mesh by taking UV from
    ## nearest neighbor on original mesh
    #simplified_verts = np.asarray(simplified_mesh.vertices)
    #corresps = []
    #orig_verts_kdtree = open3d.geometry.KDTreeFlann(np.asarray(mesh.vertices).T)
    #for k in range(simplified_verts.shape[0]):
    #    corresps.append(orig_verts_kdtree.search_knn_vector_3d(simplified_verts[k, :], 1)[2])
    ## Now go through and update triangle uvs
    #n_tris = simplified_verts.num_triangles()
    #tri_inds = np.asarray(simplified_verts.triangles)
    #triangle_uvs = np.zeros(3*n_tris, 2)
    #for k in range(n_tris):
    #    for l in range(3):
    #        corresps[tri_inds[k, l]]
    open3d.io.write_triangle_mesh(output_obj_path, simplified_mesh)

    # Hardcode creation of a texture next to it,
    # looking for the texture in ../materials/textures/texture.png
    # (Downsize the texture to a reasonable size, too)
    dirname, _ = os.path.split(input_obj_path)
    texture_path = os.path.join(dirname, "../materials/textures/texture.png")
    if os.path.exists(texture_path):
        texture_image = open3d.io.read_image(texture_path)
        texture_image = cv2.cvtColor(np.asarray(texture_image), cv2.COLOR_BGR2RGB)
        # Resize to height 1024 (which hopefully results in
        # a reasonably-sized image. 
        height, width = texture_image.shape[:2]
        if height > width:
            texture_image = imutils.resize(texture_image, height=1024)
        else:
            texture_image = imutils.resize(texture_image, width=1024)
        cv2.imwrite(output_texture_path, texture_image)


    else:
        print("Found no texture at ", texture_path)

    return output_obj_path

def do_collision_mesh_simplification(input_obj_path, show=False):
    '''
    Given an obj, performs a convex decomposition of it with
    trimesh _ vhacd, saving all the parts in a subfolder.
    
    Args:
    - input_obj_path: String path to obj file to decompose
    - show: Whether to open (and block on) a window to preview
    the decomposition.

    Returns: (out_paths, inertia)
    - out_paths: List of generated obj files
    - inertia: total inertia of the obj, assuming density of 2000 kg/m^3
    '''

    # Create a subdir for the convex decomp parts, as
    # there might be many.
    dirname, objname = os.path.split(input_obj_path)
    piece_dirname = objname[:-4] + "_parts"
    out_dir = os.path.join(dirname, piece_dirname)
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(input_obj_path)
    if show:
        mesh.show()
    try:
        convex_pieces = trimesh.decomposition.convex_decomposition(
            mesh) # TODO: args
        if not isinstance(convex_pieces, list):
            convex_pieces = [convex_pieces]
    except Exception as e:
        print("Problem in decomp: ", e)
    
    # Give them random colors for display
    for part in convex_pieces:
        this_color = trimesh.visual.random_color()
        part.visual.face_colors[:] = this_color
    scene = trimesh.scene.scene.Scene()
    for part in convex_pieces:
        scene.add_geometry(part)
    if show:
        scene.show()

    mesh.density = 2000
    I = mesh.moment_inertia
    inertia = SpatialInertia.MakeFromCentralInertia(
        mass=mesh.mass,
        p_PScm_E=mesh.center_mass,
        I_SScm_E=RotationalInertia(
            Ixx=I[0, 0],
            Ixy=I[0, 1],
            Ixz=I[0, 2],
            Iyy=I[1, 1],
            Iyz=I[1, 2],
            Izz=I[2, 2]).ShiftFromCenterOfMass(mass=mesh.mass,
                                               p_BcmQ_E=-mesh.center_mass)
    )

    # Save them each out
    out_paths = []
    for k, part in enumerate(convex_pieces):
        piece_name = '%s_convex_piece_%03d.obj' % (objname[:-4], k)
        full_path = os.path.join(out_dir, piece_name)
        trimesh.exchange.export.export_mesh(part, full_path)
        out_paths.append(full_path)

    return out_paths, inertia


def update_sdf_with_convex_decomp(input_file):
    '''
    Given an SDF from IgnitionRobotics, produces a new SDF that:
    - For each visual item, replaces it with a simplified
    visual item with a simplified texture saved alongside,
    which are auto-generated next to the original
    model.
    - For each collision item, replaces it with a convex
    decomp of the collision geometry, which is auto-generated
    next to the original model.
    - Inserts inertia for each link, calculated from the
    collision geometry with an assumed constant density.
    and saves this SDF alongside the original with a
    "_simplified.sdf" suffix.
    '''

    this_file_path, _ = os.path.split(input_file)

    # Open up the SDF as an XML
    parser = ET.XMLParser(remove_blank_text=True)
    data = ET.parse(input_file, parser=parser)
    # Add drake namespace
    ET.register_namespace('drake', 'drake.mit.edu')
    sdf_root = data.getroot()

    link_items = sdf_root.findall('.//link')

    for link_item in link_items:
        visual_items = link_item.findall('visual')
        for visual_item in visual_items:
            # Simplify each mesh
            for visual_mesh in visual_item.findall('.//mesh'):
                for uri in visual_mesh.findall('.//uri'):
                    model_path = os.path.join(this_file_path, uri.text)
                    new_model_path = do_visual_mesh_simplification(model_path)
                    new_model_path = os.path.relpath(new_model_path, this_file_path)
                    # Replace mesh reference with our simplified version
                    uri.text = new_model_path

        collision_items = link_item.findall('collision')
        component_inertias = []
        total_collision_k = 0
        for collision_item in collision_items:
            new_model_paths = []
            for geometry in collision_item.findall('.//geometry'):
                for uri in geometry.findall('.//uri'):
                    model_path = os.path.join(this_file_path, uri.text)
                    new_paths, new_inertia = do_collision_mesh_simplification(model_path)
                    new_model_paths += new_paths
                    component_inertias.append(new_inertia)
            new_model_paths = [
                os.path.relpath(new_model_path, this_file_path)
                for new_model_path in new_model_paths]
            # Create a new XML subtree for the collection of meshes
            # we just created. I *think* each convex piece needs
            # to be in its own collision tag, otherwise Drake
            # seems to be ignoring them...
            link_item.remove(collision_item)

            for path in new_model_paths:
                collision_item = ET.SubElement(link_item, 'collision', name="collision_%04d" % total_collision_k)
                total_collision_k += 1
                geometry_item = ET.SubElement(collision_item, 'geometry')
                mesh_item = ET.SubElement(geometry_item, "mesh")
                uri_item = ET.SubElement(mesh_item, "uri")
                uri_item.text = path
                ET.SubElement(mesh_item, '{drake.mit.edu}declare_convex')

        inertial_items = link_item.findall('inertial')
        if len(inertial_items) == 0:
            # Estimate inertia from the collision mesh pieces
            if len(component_inertias) != 1:
                raise NotImplementedError("Don't know how to combine SpatialInertias.")
            else:
                inertia = component_inertias[0]
            inertial = ET.SubElement(link_item, 'inertial')
            pose_item = ET.SubElement(inertial, 'pose')
            pose_item.text = "{:.2E} {:.2E} {:.2E} 0. 0. 0.".format(
                                *inertia.get_com().tolist())
            mass_item = ET.SubElement(inertial, 'mass')
            mass_item.text = '{:.2E}'.format(inertia.get_mass())
            I = [['{:.4E}'.format(y) for y in x]  # NOQA
                 for x in inertia.CalcRotationalInertia().CopyToFullMatrix3()[:]]
            ET.SubElement(
                inertial,
                'inertia',
                ixx=I[0][0],
                ixy=I[0][1],
                ixz=I[0][2],
                iyy=I[1][1],
                iyz=I[1][2],
                izz=I[2][2])


    output_path = input_file[:-4] + "_simplified.sdf"
    data.write(output_path, pretty_print=True)

if __name__ == "__main__":
    # attach to logger so trimesh messages will be printed to console
    #trimesh.util.attach_to_log()

    # Set to_update to grab the model SDFs from extracted
    # downloaded IgnitionRobotics SDFs.
    # I assume here that you've downloaded a set of model archives
    # from the site, and put them each in their own folder in a
    # data directory somewhere. Each subfolder should have a
    # model.sdf file next to "meshes" and "materials" folders.

    # e.g. the following directories exist:
    # <my data dir>/<model name>/model.sdf
    # <my data dir>/<other>/model.sdf

    data_folder = "/home/gizatt/projects/scene_grammar/models/"
    # Update a specific model by name.
    to_update = glob.glob(data_folder + "/*/Chefmate_8_Frypan/model.sdf")
    # Update all models.
    #to_update = glob.glob(data_folder + "/*/*/model.sdf")
    print(to_update)
    for file in to_update:
        print("Processing %s" % file)
        update_sdf_with_convex_decomp(file)