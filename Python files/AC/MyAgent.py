from AbstractAgent import AbstractAgent
from ACNetwork import ACNetwork
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_state_dict = torch.load("ac-5.tar", map_location="cpu")["model_state_dict"]


def format_observations(observation, keys=("glyphs", "blstats")):
    observations = {}
    for key in keys:
        entry = observation[key]
        entry = torch.from_numpy(entry)
        entry = entry.view((1, 1) + entry.shape)  # (...) -> (T,B,...).
        observations[key] = entry
    return observations

def list_to_device(l):
    l = torch.stack([ t.to(device) for t in l ])
    return l

def dict_to_device(dic):
    for key, value in dic.items():
        dic[key] = dic[key].to(device)
    return dic

class MyAgent:
    def __init__(self, observation_space, action_space):
        """Loads the agent"""
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = ACNetwork(observation_space, self.action_space.n)

        
#        _state_dict = torch.load("ac-5.tar", map_location="cpu")["model_state_dict"]
        self.model.load_state_dict(_state_dict)
        self.core_state = self.model.initial_state()


        self.model.to(device)
        self.model.eval()

    def act(self, observation):
        # Perform processing to observation

        done = torch.tensor([False]).view(1,1).float()
        state = format_observations(observation)
        with torch.no_grad():
            outputs, self.core_state = self.model(dict_to_device(state), list_to_device(self.core_state), done.to(device))
            action = (outputs['action'])
        return action.item()
