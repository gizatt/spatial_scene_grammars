Spatial Scene Grammars
-------------

<img src="examples/kitchen/demo_rooms.gif" 
alt="Preview of some generated rooms" border="10" />

Environment generation tool using attributed stochastic scene grammars.

## Dependencies

From pip or conda:
- `numpy`
- `matplotlib`
- `networkx`
- `pyro`
- `meshcat`

Not on pip yet:
- `pydrake`: Install from [here](https://drake.mit.edu/python_bindings.html).


## Personal Notes / TODOs

Grammar setup:
- Prior work is usually using attributed AND/OR/SET grammar. I'm currently being more flexible,
but this might bite me when it comes time to get tree parsing working.
- General way of specifying constraints would be nice.
- Currently demonstrating sampling with rejection sampling (in kitchen example). Next versions should be:
  - Spiral 1: HMC in constraint null space, something like [this](https://dritchie.github.io/pdf/hmc.pdf)?
  - Spiral 2: Alternate that with tree resampling / parsing?
- Need tree parsing first pass to figure out pain points in interop between the tree structure and Pyro.