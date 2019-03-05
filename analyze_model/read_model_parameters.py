import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot
from torch.utils import model_zoo


# params = model_zoo.load_url('model.pt7')

params = torch.load('model.pt7')

# convert numpy arrays to torch Variables
for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)

print('\nTotal parameters:', sum(v.numel() for v in params.values()))

