import pyro
import matplotlib.pyplot as plt
import numpy as np
import torch

# Sanity checking that I'm getting dist params correct

vals = torch.tensor(np.arange(0., 1., 0.01))

def make_dist(tightness):
    return pyro.distributions.Beta(tightness, tightness)
 
def plot_dist(dist):
    plt.plot(vals, dist.log_prob(vals).detach().cpu().numpy())


plt.figure()
for t in [1., 2., 3., 4., 5.]:
    plot_dist(make_dist(t))
plt.show()