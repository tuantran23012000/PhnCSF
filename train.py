
import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
from models.hpn import Hypernetwork
import numpy as np
import torch
from matplotlib import pyplot as plt
from create_pareto_front import PF
from methods.CSF import CS_functions
import argparse
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from tools.hv import HvMaximization
import yaml
from torch.autograd import Variable

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_legend = {'family': 'Times New Roman',
               'weight': 'normal',
                'size': 18,
               }
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def circle_points_random(r, n):
    """
    generate n random unit vectors
    """
    
    circles = []
    for r, n in zip(r, n):
        t = np.random.rand(n) * 0.5 * np.pi  
        t = np.sort(t)
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    #print(circles)
    return circles
def train_2d(device, cfg,criterion):
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    
    loss1 = f_1
    loss2 = f_2
    
    start = time.time()
        
    sol = []

    hnet = Hypernetwork(ray_hidden_dim = ray_hidden_dim, out_dim = out_dim, n_tasks = n_tasks,num_hidden_layer=num_hidden_layer,last_activation=last_activation)
    hnet = hnet.to(device)
    if type_opt == 'adam':
        optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd)
    
    #epoch_iter = trange(epochs)
    for epoch in tqdm(range(epochs)):
        dem += 1

        hnet.train()
        optimizer.zero_grad()

        loss_torch_per_sample = []
        loss_numpy_per_sample = []
        loss_per_sample = []
        outputs = []
        rays = []
        penalty = []
        dynamic_weights_per_sample = None
        head = None
        partition = None
        n_mo_sol = None
        n_mo_obj = None
        mo_opt = None
        if criterion == 'HVI':
            head = cfg['TRAIN']['Solver'][criterion]['Head']
            partition = np.array([1/head]*head)
            n_mo_sol = cfg['TRAIN']['Solver'][criterion]['Head']
            n_mo_obj = cfg['TRAIN']['N_task']
            start = 0.
            end = np.pi/2
            mo_opt = HvMaximization(n_mo_sol, n_mo_obj, ref_point)
            
            dem = 0
            for i in range(n_mo_sol):
    
                ray = torch.from_numpy(
                    np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
                    ).to(device)
                rays.append(ray)
                output = hnet(rays[i])
                
                outputs.append(output)

                #outputs.append(net_list[i](weights[i])[0])
                
                loss_per_sample = torch.stack([loss1(outputs[i]), loss2(outputs[i])])

                loss_torch_per_sample.append(loss_per_sample)
                loss_numpy_per_sample.append(loss_per_sample.cpu().detach().numpy())

                penalty.append(torch.sum(loss_torch_per_sample[i]*rays[i])/
                                (torch.norm(loss_torch_per_sample[i])*torch.norm(rays[i])))

            loss_numpy_per_sample = np.array(loss_numpy_per_sample)[np.newaxis, :, :].transpose(0, 2, 1) #n_samples, obj, sol

            n_samples = 1
            dynamic_weights_per_sample = torch.ones(n_mo_sol, n_mo_obj, n_samples)
        else:
            ray = torch.from_numpy(
                np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32).flatten()
            ).to(device)
            hnet.train()
            optimizer.zero_grad()
            output = hnet(ray)
            ray_cs = 1/ray
            ray = ray.squeeze(0)
            l1 = loss1(output)
            l2 = loss2(output)
            losses = torch.stack((l1, l2)) 
            CS_func = CS_functions(losses,ray)
        loss = CS_func.get_criterion(criterion = criterion,rho = cfg['TRAIN']['Solver'][criterion]['Rho'],dynamic_weights_per_sample = dynamic_weights_per_sample,\
            ub = cfg['TRAIN']['Solver'][criterion]['Ub'],ray_cs = ray_cs,n_params = count_parameters(hnet),parameters = hnet.parameters(),penalty=penalty,partition=partition,head=head,mo_opt=mo_opt)
        loss.backward()
        optimizer.step()
        sol.append(output.cpu().detach().numpy().tolist()[0])
    end = time.time()
    time_training = end-start
    torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+".pt"))
    return sol,time_training
def main(cfg,criterion,device,cpf):
    if cfg['MODE'] == '2d':
        if cfg['NAME'] == 'ex1':
            
            pf = cpf.create_pf5() 
        else:
            pf = cpf.create_pf6() 
        sol, time_training = train_2d(device,cfg,criterion)
        print("Time: ",time_training)  
        draw_2d(sol,pf,cfg,criterion)
    else:
        pf  = cpf.create_pf_3d()
        sol, time_training = train_3d(device,cfg,criterion)
        print("Time: ",time_training)  
        draw_3d(sol,pf,cfg,criterion)

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/ex1.yaml', help="config file")
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","Cheby","Utility","Cosine","Cauchy","Prod","Log","AC","MC","HV","CPMTL","EPO","HVI"], default="Utility", help="solver"
    )
    args = parser.parse_args()
    criterion = args.solver 
    config_file = args.config

    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    
    if cfg['NAME'] == 'ex1':
        from problems.pb1 import f_1, f_2
        cpf = PF(f_1, f_2,None)
    elif cfg['NAME'] == 'ex2':
        from problems.pb2 import f_1, f_2
        cpf = PF(f_1, f_2,None)
    if cfg['NAME'] == 'ex3':
        from problems.pb3 import f_1, f_2, f_3
        cpf = PF(f_1, f_2, f_3)
    main(cfg,criterion,device,cpf)