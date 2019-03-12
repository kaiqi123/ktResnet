import torch

def tr(v):
    if v.ndim == 4:
        return v.transpose(2, 3, 1, 0)
    elif v.ndim == 2:
        return v.transpose()
    return v

params = {k: v.detach().cpu().numpy() for k, v in torch.load('cifar10_input/model_d28w10.pt7')['params'].items()}
params = {k: tr(v) for k, v in params.items()}

for k, v in sorted(params.items()):
    print(k, tuple(v.shape))

"""
for k, v in sorted(params.items()):
    if 'bn' in k:
        params_new[k] = v
        print(k, params_new[k].shape)
    else:
        params_new[k] = tr(v)
        print(k, params_new[k].shape)
print("---------------------")
params = params_new
"""
