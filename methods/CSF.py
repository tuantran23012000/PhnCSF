
import torch
from EPO import EPOSolver
class CS_functions():
    def __init__(self,losses,ray):
        super().__init__()
        self.losses = losses
        self.ray = ray
    def linear_function(self):
        ls = (self.losses * self.ray).sum()
        return ls

    def log_function(self):
        return (self.ray*torch.log(self.losses+1)).sum()

    def ac_function(self,rho):
        ls = (self.losses * self.ray).sum()
        cheby = max(self.losses * self.ray)
        return cheby + rho*ls
    
    def mc_function(self,rho):
        ls = (self.losses * self.ray).sum()
        cheby = max(self.losses * self.ray + rho*ls)
        return cheby
    
    def hvi_function(self,dynamic_weights_per_sample,rho,penalty,partition,head,mo_opt):
        n_samples = 1
        # dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
        for i_sample in range(0, n_samples):
            weights_task = mo_opt.compute_weights(self.losses[i_sample,:,:])
            dynamic_weights_per_sample[:, :, i_sample] = weights_task.permute(1,0)
        
        # dynamic_weights_per_sample = dynamic_weights_per_sample.to(device)
        i_mo_sol = 0
        total_dynamic_loss = torch.mean(torch.sum(dynamic_weights_per_sample[i_mo_sol, :, :]
                                                * self.losses[i_mo_sol], dim=0))
        
        for idx in range(head):
            total_dynamic_loss -= rho*penalty[idx]*partition[idx]
            
            
        hvi = total_dynamic_loss/head
        return hvi

    def product_function(self):
        return torch.prod((self.losses+1)**self.ray)

    def cosine_function(self):
        rl =self.losses * self.ray
        l_s = torch.sqrt((self.losses**2).sum())
        r_s = torch.sqrt((self.ray**2).sum())
        cosine = - (rl.sum()) / (l_s*r_s)
        return cosine

    def utility_function(self,ub):
        
        U = 1/torch.prod((ub - self.losses)**self.ray)
        return U

    def chebyshev_function(self):
        cheby = max(self.losses * self.ray)
        return cheby

    def kl_function(self):
        m = len(self.losses)
        rl = torch.exp(self.losses * self.ray)
        normalized_rl = rl / (rl.sum())
        KL = (normalized_rl * torch.log(normalized_rl * m)).sum() 
        return KL

    def cauchy_schwarz_function(self):
        rl = self.losses * self.ray
        l_s = (self.losses**2).sum()
        r_s = (self.ray**2).sum()
        cauchy_schwarz = 1 - ((rl.sum())**2 / (l_s*r_s))
        return cauchy_schwarz
    
    def get_criterion(self, criterion,rho = None,dynamic_weights_per_sample = None,ub = None,ray_cs = None,n_params = None,parameters = None,penalty=None,partition=None,head=None,mo_opt=None):
        if criterion == 'Prod':
            return self.product_function()
        elif criterion == 'Log':
            return self.log_function()
        elif criterion == 'AC':
            return self.ac_function(rho = rho)
        elif criterion == 'MC':
            return self.mc_function(rho = rho)
        elif criterion == 'HVI':
            return self.hvi_function(dynamic_weights_per_sample,rho,penalty,partition,head,mo_opt)
        elif criterion == 'LS':
            return self.linear_function()
        elif criterion == 'Cheby':
            return self.chebyshev_function()
        elif criterion == 'Utility':
            return self.utility_function(ub = ub)
        elif criterion == 'KL':
            return self.kl_function()
        elif criterion == 'Cosine':
            return self.cosine_function()
        elif criterion == 'Cauchy':
            self.ray = ray_cs
            return self.cauchy_schwarz_function()
        elif criterion == 'EPO':
            solver = EPOSolver(n_tasks=2, n_params=n_params)
            return solver(self.losses, self.ray, list(parameters))

    
