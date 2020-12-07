import torch

ch = torch.load("ac2-5.tar", map_location="cpu")
ch = {'model_state_dict':ch['model_state_dict']}
torch.save(ch, 'dqn-4.tar')
