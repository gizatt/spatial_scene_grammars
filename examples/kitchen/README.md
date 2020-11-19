Kitchen Example
------------------

<img src="demo_rooms.gif" 
alt="Preview of some generated rooms" border="10" />

# Running a demo

In one terminal, run `meshcat-server`, and open a browser to the specified
meshcat view url (usually http://127.0.0.1:7000/static/).

In another terminal, invoke `python3 run_grammar.py` with no arguments. A scene
should be generated and visualized in the meshcat browser window.

# Grammar overview

- A single kitchen generates from a `Kitchen` node, with random
length and width.
- 4 `Wall` nodes and a `Floor` spawn deterministically in the
right locations (2 walls are currently disabled for visualization).
- The `Floor` produces a `Table` at [1., 0., 0.] and a `RobotSpawnLocation`
at [-1., 0., 0.].
- Each `Table` , creates a `PlanarObjectRegion` on its surface.
- Each `Wall` can spawn a geometric number of `Cabinets` on its surface.
- Each `Cabinet` spawns with its doors in random positions, and creates
3 levels of objects on each of its shelves, represented by `PlanarObjectRegion` nodes.
- Each `PlanarObjectRegion` can produce a geometric number of `Object`s (random
x, y, yaw; fixed z, roll, pitch) within the volume it was constructed with.
- Right now, `Object` is a terminal node with a box for geometry.
- `RobotSpawnLocation` is a terminal node (indicating where to put the robot).

All nodes have poses, and most (excluding `RobotSpawnLocation` and `PlanarObjectRegion`)
have visual and collision geometry as well. `Wall`, `Floor`, `Cabinet`, `Table`, and
`RobotSpawnLocation` have additional clearance geometry specifications so we can check
if the room configuration is feasible w.r.t. the furniture all having reasonable clearance.

# File overview
- `run_grammar.py`: Top-level / entrypoint. Currently contains a routine
for setting up rejection sampling kitchen scenes that satisfy some hand-written
nonpenetration and object count constraints; and code for setting up a
visualization of the scene with a PR2 inserted for scale.
- `grammar_room_layout.py`: Implements `Kitchen` (root), `Wall`, `Floor`, and `RobotSpawnLocation` nodes.
- `grammar_table.py`: Implemented `Table` node.
- `grammar_cabinet.py`: Implemented `Cabinet` node.
- `grammar_objects.py`:  Implements `Object` and `PlanarObjectRegion` nodes.
