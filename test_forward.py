import torch
from proposed_method import PointCloudSimplifier

model  = PointCloudSimplifier(M=512)
P      = torch.randn(2, 1024, 3)   # batch=2, N=1024 points

out    = model(P)
print(out['P_simplified'].shape)   # → torch.Size([2, 512, 3])
print(out['P_recon'].shape)        # → torch.Size([2, 512, 3])
print(out['loss']['total'])        # → scalar tensor