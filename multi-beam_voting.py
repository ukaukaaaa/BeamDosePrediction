import torch

def reconstruct(inputs, coarse_dose):
    """
    inputs: b,9,128,128,128
    caorse_dose: b,1,128,128,128
    """
    a = torch.sum(inputs>0, dim=1).unsqueeze(1)

    out = torch.zeros_like(coarse_dose).float()
    out[torch.where(a == 0)] = coarse_dose[torch.where(a == 0)].float()
    out[torch.where(a != 0)] = (torch.sum(inputs, dim=1).unsqueeze(1)[torch.where(a != 0)] / a[torch.where(a != 0)]).float()
    return out
