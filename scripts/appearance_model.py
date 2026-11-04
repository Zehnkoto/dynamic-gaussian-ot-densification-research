"""Time-shared appearance residuals for fixed-geometry distillation."""
import torch
class Appearance(torch.nn.Module):
    def __init__(self,sh,logits,k,mode='joint'):
        super().__init__()
        self.register_buffer('source_sh',sh.detach().clone())
        self.register_buffer('source_logits',logits.detach().clone())
        self.dc=torch.nn.Parameter(torch.zeros(k,1,3,device=sh.device))
        self.rest=torch.nn.Parameter(torch.zeros(k,sh.shape[1]-1,3,device=sh.device))
        self.opacity=torch.nn.Parameter(torch.zeros(k,1,device=sh.device))
        if mode not in ['joint','sh','opacity']:raise ValueError(mode)
        self.dc.requires_grad_(mode!='opacity');self.rest.requires_grad_(mode!='opacity');self.opacity.requires_grad_(mode!='sh')
    def forward(self,ids):
        return self.source_sh[ids]+torch.cat([self.dc,self.rest],1),torch.sigmoid(self.source_logits[ids]+self.opacity)
    def optimizer(self,rates):
        return torch.optim.Adam([{'params':[getattr(self,n)],'lr':v} for n,v in rates.items() if getattr(self,n).requires_grad],eps=1e-8)
