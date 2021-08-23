from simple_3d_ssg import *

''' Minimal test grammar for testing the
rotation-around-axis-within-bounds constraint. '''

class RotationTestRoot(AndNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=RotationTestIntermediate,
                child_xyz_bounds=XyzBounds(np.array([0.1, 0.1, 0.1]), np.array([0.9, 0.9, 0.9])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), np.pi/2., np.pi/2)
            )
        ]
        super().__init__(
            child_infos=child_infos,
            tf=tf,
            geometry=None,
            observed=False
        )

class RotationTestIntermediate(AndNode):
    def __init__(self, tf):
        child_infos = [
            ChildInfo(
                child_type=RotationTestTerminal,
                child_xyz_bounds=XyzBounds(np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])),
                child_rotation_bounds=RotationBounds(np.array([0., 0., 1.]), -np.pi/10, np.pi/10)
            )
        ]
        super().__init__(
            child_infos=child_infos,
            tf=tf,
            geometry=None,
            observed=False
        )
        
class RotationTestTerminal(TerminalNode):
    def __init__(self, tf):
        super().__init__(
            tf=tf,
            geometry=meshcat_geom.Box([0.025, 0.025, 0.1]),
            geometry_color=0xaa6622,
            observed=True
        )