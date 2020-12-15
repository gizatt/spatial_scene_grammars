import unittest

from pydrake.all import (
    PackageMap
)

from spatial_scene_grammars.drake_interop import *
from spatial_scene_grammars.serialization_model_directive import *

class PackageHandlingTest(unittest.TestCase):
    def test_package_uri_resolving(self):
        package_map = PackageMap()
        # Make a dummy package
        make_default_package_xml("test_package", "/tmp/pkg/test_package/package.xml")
        # See if we can recover the package from ROS_PACKAGE_PATH.
        if "ROS_PACKAGE_PATH" not in os.environ.keys():
            os.environ["ROS_PACKAGE_PATH"] = ""
        os.environ["ROS_PACKAGE_PATH"] += ":/tmp/pkg/"
        package_map.PopulateFromEnvironment("ROS_PACKAGE_PATH")
        
        full_path = "/tmp/pkg/test_package/fake_file"
        test_input = "test_package://fake_file"
        resolved_path = resolve_catkin_package_path(package_map, test_input)
        with self.subTest("Resolving a package path:"):
            self.assertEqual(resolved_path, full_path)
        resolved_path = resolve_catkin_package_path(package_map, full_path)
        with self.subTest("Resolving a non-package path:"):
            self.assertEqual(resolved_path, full_path)

if __name__ == "__main__":
    unittest.main()