import re
import torch
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchviz import make_dot
# from torch.utils import model_zoo
import numpy as np
import pickle

checkpoint = torch.load('model.pt7')

print(checkpoint.keys())

params_pytorch = checkpoint['params']

#np.save('model_w10d28.npy', params_pytorch)
#params = np.load('model_w10d28.npy')
#params = dict(params)

with open('model_w10d28.pkl', 'wb') as f:
    pickle.dump(params_pytorch, f, pickle.HIGHEST_PROTOCOL)

with open('model_w10d28.pkl', 'rb') as f:
    params = pickle.load(f)


for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)

print('\nTotal parameters:', sum(v.numel() for v in params.values()))


"""
# params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')

# convert numpy arrays to torch Variables
for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)

print('\nTotal parameters:', sum(v.numel() for v in params.values()))
"""

