import numpy as np

import random

from torchsummary import summary

import argparse

import torch.nn as nn
import torch

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import write_video

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import os
from pdb import set_trace as trace
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--num_batches", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_actions", type=int, default=6, help="number of attributes for dataset")
parser.add_argument("--max_num_items", type=int, default=50000, help="number of attributes for dataset")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval betwen image samples")

parser.add_argument('--gen_ckpt', help='generator checkpoint')
parser.add_argument('--disc_ckpt', help='discriminator checkpoint')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
from robogym.envs.rearrange.blocks_reach import make_env
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()



        self.backbone = nn.Sequential(
                            nn.Linear(96, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                         )
        self.a1 = nn.Sequential( nn.Linear(128, 11), nn.Softmax())
        self.a2 = nn.Sequential( nn.Linear(128, 11), nn.Softmax())
        self.a3 = nn.Sequential( nn.Linear(128, 11), nn.Softmax())
        self.a4 = nn.Sequential( nn.Linear(128, 11), nn.Softmax())
        self.a5 = nn.Sequential( nn.Linear(128, 11), nn.Softmax())
        self.a6 = nn.Sequential( nn.Linear(128, 11), nn.Softmax())

    def forward(self, x):
        x = self.backbone(x)
        a1 = self.a1(x)
        a2 = self.a2(x)
        a3 = self.a3(x)
        a4 = self.a4(x)
        a5 = self.a5(x)
        a6 = self.a6(x)
        
        return a1,a2,a3,a4,a5,a6
env = make_env(
    constants={
        'randomize': True,
        'mujoco_substeps': 10,
        'max_timesteps_per_goal': 400
    },
    parameters={
        'n_random_initial_steps': 100,
    }
)
def get_observation(obs):
     
    """
     
    :returns
    """
    all_obs = obs.copy()
    keys = sorted(all_obs.keys())
    obs_vec = []
    for key in keys: 
        if key != "safety_stop"  and key != "is_goal_achieved" :
            if not isinstance(all_obs[key], list):  
                if np.prod(all_obs[key].shape) == 1 : 
                    obs_vec += all_obs[key].tolist()
                else: 
                    obs_vec += all_obs[key].squeeze().tolist()
                 
            else: obs_vec += all_obs[key]
             
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#    obs_vec = FloatTensor(torch.from_numpy(np.array([obs_vec]).astype(np.float32)).cuda())
    obs_vec = np.array([obs_vec]).astype(np.float32)
    return obs_vec
def get_new_goal(obs1,obs2): 
         
    """
     
    :returns
    """
    obs = obs1.copy()
    obs["goal_obj_pos"] = obs2["obj_pos"]
    obs["goal_obj_rot"]=obs2["obj_rot"]
    obs["goal_rel_obj_pos"]=  obs2["rel_goal_obj_pos"]
    return obs
def add_to_buffer(my_buffer, item,max_num_items): 
     
    """
     
    :returns
    """
    my_buffer.insert(0, item)
    if len(my_buffer) > max_num_items: my_buffer.pop()
    
def print_obs(obs): 
     
    """
     
    :returns
    """
    shape_sum = 0
    keys = sorted(obs.keys())
    for idx, key in enumerate(keys):
        print(f"{key} is {obs[key].shape } , {obs[key]}, idx:  {idx} ")
        shape_sum += np.prod(obs[key].shape)
po = print_obs
actor = Actor()
summary(actor, (96,))
observation_buffer = []
shape_sum = 0
action_loss = torch.nn.CrossEntropyLoss()
if cuda:
    print(f"cuda is {cuda } ")
    action_loss.cuda()
    actor.cuda()
al = action_loss
optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
LT = LongTensor
FT = FloatTensor

#print_obs(obs_list)
my_buffer = []
#for epoch in range(opt.n_epochs):
epoch = 0
while True:
    if epoch % 10 == 0: 
        print(f"epoch is {epoch } ")
    epoch += 1
    obs_list = env.reset()
    wdx = 0
    while True:
        wdx += 1
        if wdx % 50 == 0: 
            print(f"len(my_buffer) is {len(my_buffer) } ")
    
        obs_list["goal_rel_obj_pos"] = np.array([0,0,0]  )
        obs_npy = get_observation(obs_list)
#        with torch.no_grad():
#            a1,a2,a3,a4,a5,a6 = actor(obs_npy)
#            a1 = torch.argmax(a1).cpu().numpy()
#            a2 = torch.argmax(a2).cpu().numpy()
#            a3 = torch.argmax(a3).cpu().numpy()
#            a4 = torch.argmax(a4).cpu().numpy()
#            a5 = torch.argmax(a5).cpu().numpy()
#            a6 = torch.argmax(a6).cpu().numpy()
#            action = [int(a1),int(a2),int(a3),int(a4),int(a5),int(a6)]
            
        action = env.action_space.sample()
        new_obs_list,_,terminal,_ = env.step(action)    
    #    new_obs_npy = get_observation(new_obs_list)
        goal = get_new_goal(obs_list, new_obs_list)
        goal_npy = get_observation(goal)
#        a1,a2,a3,a4,a5,a6 = actor(goal_npy)
        p1,p2,p3,p4,p5,p6 = action 
#        print(f"action is {action } ")
#        p1,p2,p3,p4,p5,p6 = LT([p1]),LT([p2]),LT([p3]),LT([p4]),LT([p5]),LT([p6]) 
        add_to_buffer(my_buffer, [goal_npy, action], opt.max_num_items)
           
        obs_list = new_obs_list
        if terminal: 
            print(f"terminal is {terminal } ")
            break 
    if len(my_buffer) == 50000:
        print(f"len(my_buffer) is {len(my_buffer) } ")
        for this_batch in range(opt.num_batches):
            batch =  random.sample(my_buffer, opt.batch_size)
            Goals = FT(torch.from_numpy(np.array([sample[0] for sample in batch]).squeeze()).cuda())
            Actions = [sample[1] for sample in batch]
            p1 = LT(np.array([action[0] for action in Actions]))
            p2 = LT(np.array([action[1] for action in Actions]))
            p3 = LT(np.array([action[2] for action in Actions]))
            p4 = LT(np.array([action[3] for action in Actions]))
            p5 = LT(np.array([action[4] for action in Actions]))
            p6 = LT(np.array([action[5] for action in Actions]))
            a1,a2,a3,a4,a5,a6 = actor(Goals)
            a1loss = al(a1,p1)
    #        print(f"p1 is {p1 } ")
    #        print(f"a1 is {a1 } ")
    #        print(f"a1loss is {a1loss } ")
            a2loss = al(a2,p2) 
            a3loss = al(a3,p3) 
            a4loss = al(a4,p4) 
            a5loss = al(a5,p5) 
            a6loss = al(a6,p6)
            act_loss_total = a1loss + a2loss + a3loss + a4loss + a5loss + a6loss
            print(f"act_loss_total is {act_loss_total } ")
            act_loss_total.backward()
            optimizer.step()
    

    
    
    
#    print('\n\n\n')
#    print(f"new_obs_list is")
#    print_obs(new_obs_list)
#    print('\n\n\n')
#    print(f"new obs is")
#    print_obs(gt)
    
   
     
     
      

print(f"shape_sum is {shape_sum } ")
    

