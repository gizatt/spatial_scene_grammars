from simple_3d_ssg import *

''' Root creates geometric number of oriented
clusters in the unit box. Each cluster produces
a geometric number of long boxes with minor rotations
around their non-long axes. '''

class OrientedClusterRoot(GeometricSetNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=OrientedCluster,
                child_xyz_bounds=XyzBounds(np.array([0., 0., 0.]), np.array([1., 1., 1.])),
                child_rotation_bounds=None
            )
        ]
        super().__init__(
            child_infos=child_infos,
            tf=tf,
            p=0.3,
            max_children=2,
            geometry=None,
            observed=False
        )

class OrientedCluster(GeometricSetNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=LongBox,
                child_xyz_bounds=XyzBounds(np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])),
                child_rotation_bounds=RotationBounds(np.array([1., 0., 0.]), -np.pi/8, np.pi/8)
            )
        ]
        super().__init__(
            child_infos=child_infos,
            tf=tf,
            p=0.3,
            max_children=5,
            geometry=None,
            observed=False
        )

class BoxPreYRotation(AndNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=LongBox,
                child_xyz_bounds=XyzBounds(np.zeros(3), np.zeros(3)),
                child_rotation_bounds=RotationBounds(np.array([0., 1., 0.]), -np.pi/8, np.pi/8)
            )
        ]
        super().__init__(
            child_infos=child_infos,
            tf=tf,
            geometry=None,
            observed=False
        )

class LongBox(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            geometry=meshcat_geom.Box([0.025, 0.025, 0.1]),
            geometry_color=0xaa6622,
            observed=True
        )