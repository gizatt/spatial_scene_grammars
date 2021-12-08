from simple_3d_ssg import *

class Sink(AndNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=Plates,
                child_xyz_bounds=XyzBounds(np.array([0., 0., 0.]), np.array([0., 0., 0.])),
                child_rotation_bounds=RotationBounds(np.array([1., 0., 0.]), 0., 0.)
            ),
            ChildInfo(
                child_type=Mugs,
                child_xyz_bounds=XyzBounds(np.array([0., 0., 0.]), np.array([0., 0., 0.])),
                child_rotation_bounds=RotationBounds(np.array([1., 0., 0.]), 0., 0.)
            ),
           # ChildInfo(
           #     child_type=Utensils,
           #     child_xyz_bounds=XyzBounds(np.array([0., 0., 0.]), np.array([0., 0., 0.])),
           #     child_rotation_bounds=RotationBounds(np.array([1., 0., 0.]), 0., 0.)
           # )
        ]
        geometry = meshcat_geom.Box([1.0, 1.0, 0.1])
        geometry_tf = RigidTransform(p=np.array([0.5, 0.5, -0.05]))
        geometry_color = 0x333333
        super().__init__(
            child_infos=child_infos,
            tf=tf,
            geometry=geometry,
            geometry_tf=geometry_tf,
            geometry_color=geometry_color,
            observed=True
        )

class Plates(RepeatingSetNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=Plate,
                child_xyz_bounds=XyzBounds(np.array([0.1, 0.1, 0.0]), np.array([0.9, 0.9, 0.0])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), -np.pi, np.pi)
            )
        ]
        geometry = None
        super().__init__(
            child_infos=child_infos,
            p=0.3,
            max_children=3,
            tf=tf,
            geometry=geometry,
            observed=False
        )

class Plate(OrNode):
    def __init__(self, tf):
        # Either spawns another plate on top, or nothing.
        child_infos = [
            ChildInfo(
                child_type=Plate,
                child_xyz_bounds=XyzBounds(np.array([-0.01, -0.01, 0.03]), np.array([0.01, 0.01, 0.03])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), -np.pi, np.pi)
            ),
            ChildInfo(
                child_type=Null,
                child_xyz_bounds=XyzBounds(np.zeros(3), np.zeros(3)),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), 0., 0.)
            ),
        ]
        child_probs = np.array([0.5, 0.5])
        #geometry = meshcat_geom.Box([0.2, 0.2, 0.1])
        #geometry_tf = RigidTransform(p=np.array([0., 0., 0.05]))
        #geometry_scale = np.ones(3)
        geometry = meshcat_geom.ObjMeshGeometry.from_file("plate_11in.obj")
        geometry_scale = np.array([0.001, 0.001, 0.001])
        geometry_tf = RigidTransform(p=np.array([0., 0., 0.025]), rpy=RollPitchYaw(np.pi/2., 0., 0.))
        super().__init__(
            child_infos=child_infos,
            child_probs=child_probs,
            tf=tf,
            geometry=geometry,
            geometry_scale=geometry_scale,
            geometry_tf=geometry_tf,
            geometry_color=0xaa9933,
            observed=True
        )

class Mugs(RepeatingSetNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=Mug,
                child_xyz_bounds=XyzBounds(np.array([0.1, 0.1, 0.0]), np.array([0.9, 0.9, 0.0])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), -np.pi, np.pi)
            )
        ]
        geometry = None
        super().__init__(
            child_infos=child_infos,
            p=0.3,
            max_children=5,
            tf=tf,
            geometry=geometry,
            observed=False
        )

class Mug(OrNode):
    def __init__(self, tf):
        # May spawn some utensils inside
        child_infos = [
            ChildInfo(
                child_type=UtensilsInMug,
                child_xyz_bounds=XyzBounds(np.array([-0.0, -0.00, 0.0]), np.array([0.00, 0.00, 0.0])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), -np.pi, np.pi)
            ),
            ChildInfo(
                child_type=Null,
                child_xyz_bounds=XyzBounds(np.zeros(3), np.zeros(3)),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), 0., 0.)
            ),
        ]
        child_probs = np.array([0.5, 0.5])
        geometry = meshcat_geom.ObjMeshGeometry.from_file("mug_visual.obj")
        geometry_tf = RigidTransform(p=np.array([0., 0., 0.05]))
        super().__init__(
            child_infos=child_infos,
            child_probs=child_probs,
            tf=tf,
            geometry=geometry,
            geometry_tf=geometry_tf,
            geometry_color=0xaaff00,
            observed=True
        ) 

class Utensils(RepeatingSetNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=Utensil,
                child_xyz_bounds=XyzBounds(np.array([0.1, 0.1, 0.0]), np.array([0.9, 0.9, 0.0])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), -np.pi, np.pi)
            )
        ]
        geometry = None
        super().__init__(
            child_infos=child_infos,
            p=0.3,
            max_children=5,
            tf=tf,
            geometry=geometry,
            observed=False
        )

class UtensilsInMug(RepeatingSetNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=Utensil,
                child_xyz_bounds=XyzBounds(np.array([-0.01, -0.01, 0.04]), np.array([0.01, 0.01, 0.06])),
                child_rotation_bounds=RotationBounds(np.array([1., 0., 0.]), -np.pi/8, np.pi/8)
            )
        ]
        geometry = None
        super().__init__(
            child_infos=child_infos,
            p=0.4,
            max_children=2,
            tf=tf,
            geometry=geometry,
            observed=False
        )

class Utensil(TerminalNode):
    def __init__(self, tf):
        geometry = meshcat_geom.Box([0.02, 0.01, 0.1])
        geometry_tf = RigidTransform(p=np.array([0., 0., 0.05]))
        super().__init__(
            tf=tf,
            geometry=geometry,
            geometry_tf=geometry_tf,
            geometry_color=0x1188bb,
            observed=True
        ) 
                                                          
class Null(TerminalNode):
    def __init__(self, tf):
        geometry = None
        super().__init__(
            tf=tf,
            geometry=geometry,
            observed=False
        )