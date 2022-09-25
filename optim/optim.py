import torch.optim as optim

def adam_optim(params,
               lr,
               weight_decay):
    optimizer = optim.Adam(params,lr,weight_decay)
    return optimizer