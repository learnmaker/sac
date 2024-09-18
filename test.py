

import torch
states = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

states = torch.tensor(states, dtype=torch.float32)

# Split states into local and global parts
num_agents = states.shape[0]
print(num_agents)
local_states = states[range(num_agents), range(num_agents)]  # Local state for each agent
print(local_states)
global_states = torch.stack([torch.cat((states[i][:i], states[i][i+1:])) for i in range(num_agents)], dim=0)  # Global state excluding self
print(global_states)
# Reshape global states if needed
# global_states = global_states.view(num_agents, num_agents - 1, local_info_num)