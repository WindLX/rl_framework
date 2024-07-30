import torch
from torch.distributions import Normal, Categorical, Independent, MixtureSameFamily
import matplotlib.pyplot as plt
import numpy as np

# Setting parameters
batch_size = 1
action_space = 4
num_mixtures = 2

# Creating GMM distribution
# weights = torch.rand(batch_size, num_mixtures)
weights = torch.tensor([[0.5, 0.5]])
# loc = torch.rand(batch_size, num_mixtures, action_space)
loc = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [0.0, -10.0, -2.0, -3.0]]])
# scale = torch.rand(batch_size, num_mixtures, action_space)
scale = torch.tensor([[[1.0, 3.0, 5.0, 10.0], [1.0, 3.0, 5.0, 10.0]]])
mix = Categorical(weights)
comp = Independent(Normal(loc, scale), 1)
gmm = MixtureSameFamily(mix, comp)

# Sampling from the GMM
samples = gmm.sample((10000,)).squeeze().numpy()

# Plotting
fig, axes = plt.subplots(1, action_space, figsize=(20, 5))
for i in range(action_space):
    axes[i].hist(samples[:, i], bins=100, density=True, alpha=0.6, color="g")
    axes[i].set_title(f"Action {i+1}")
plt.show()
