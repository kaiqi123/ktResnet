import re
import torch
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchviz import make_dot
# from torch.utils import model_zoo
import numpy as np
import pickle

checkpoint = torch.load('model_d28w10.pt7')
print(checkpoint.keys())
params = checkpoint['params']

for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)
print('\nTotal parameters:', sum(v.numel() for v in params.values()))

"""
with open('model_w10d28.pkl', 'wb') as f:
    pickle.dump(params_pytorch, f, pickle.HIGHEST_PROTOCOL)

with open('model_w10d28.pkl', 'rb') as f:
    params = pickle.load(f)
"""


"""
# params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')

# convert numpy arrays to torch Variables
for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)

print('\nTotal parameters:', sum(v.numel() for v in params.values()))
"""

