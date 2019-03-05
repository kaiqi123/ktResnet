import re
import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torchviz import make_dot
# from torch.utils import model_zoo

def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    #for i, (key, v) in enumerate(params.items()):
    for i, (key, v) in enumerate(sorted(params.items())):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


checkpoint = torch.load('model.pt7')

print(type(checkpoint))
print(checkpoint.keys())
#print(type(params.items()))
#print_tensor_dict(params)


"""
# params = model_zoo.load_url('https://s3.amazonaws.com/modelzoo-networks/wide-resnet-50-2-export-5ae25d50.pth')

# convert numpy arrays to torch Variables
for k, v in sorted(params.items()):
    print(k, tuple(v.shape))
    params[k] = Variable(v, requires_grad=True)

print('\nTotal parameters:', sum(v.numel() for v in params.values()))
"""

