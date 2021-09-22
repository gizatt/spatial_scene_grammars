Assets for Scenes
-----------------

As a tool for collecting assets for building scenes, I've written a small
pipeline script for converting SDFs to make them simulate-able by Drake, and
visualize-able by Meshcat / Drake visualizer. It assumes assets were downloaded
from the [IgnitionRobotics collection](https://app.ignitionrobotics.org/dashboard) --
even more specifically, I [boldly] assume that the model comes in with this file
structure exactly:

```
<model_folder_name>
├── model.sdf
├── meshes
│       ├── model.obj
├── materials
│       ├── textures
│              ├── texture.png
```

If the model has an mtl, it's ignored; it's assumed that the model UVs are set up
such that `texture.png` is a good baked texture for that object.

Running the `do_model_updates.py` script (you can change its target model folders
by editing the script at the bottom) will open each `model.sdf` and create a new
`model_simplified.sdf` next to it that references automatically-created altered
models and objects. No original files are altered -- new ones are made next to them.

This whole pipeline is subject to significant change very soon -- probably to make
it create ROS-package-wrapped model data folders rather than mixing in the newly
created assets with the originals.

# Software prereqs

Install deps with `pip install trimesh open3d opencv-python imutils`.

You'll need a backend that can do convex decomposition, too -- IIRC, having
`blender` on your path *might* be enough. But I usually copy the code out of
[this setup file from trimesh](https://github.com/mikedh/trimesh/blob/master/docker/builds/vhacd.bash)
and download a `VHACD` binary and just install it to my system path.
