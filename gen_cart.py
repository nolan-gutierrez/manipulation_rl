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
import gym
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
###############################################################################
### https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
### https://towardsdatascience.com/the-definitive-way-to-deal-with-continuous-variables-in-machine-learning-edb5472a2538 
###############################################################################

cuda = True if torch.cuda.is_available() else False
ENV_NAME = "CartPole-v1"
def get_iteration_noise( num_samples, dimension):
    return Variable(T.FloatTensor(num_samples, self.dim_z_motion).normal_())
class Path_Generator(nn.Module):
    def __init__(self, video_length, observation_dim):
        """
         
        :returns
        """
        super(Path_Generator, self).__init__()

        self.video_length =  video_length
        self.observation_dim = observation_dim

        self.backbone = nn.Sequential(
                            nn.Linear(68, 128), 
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
        self.emb = nn.Embedding(50, 64)
        self.recurrent = nn.GRUCell(4, 4)
    def forward(self, x, labels):
        x = torch.cat((x, self.emb(labels), 1)
        x = self.backbone(x)
        video_prediction = []
        for i in range(self.video_length):
            x = self.recurrent(get_iteration_noise(x.size(0),self.observation_dim) ,hx)
            video_prediction.append(x)
        return video_prediction
class Actor(nn.Module):
    def __init__(self, video_length, observation_dim):
        """
         
        :returns
        """
        super(Actor, self).__init__()

        self.video_length =  video_length
        self.observation_dim = observation_dim

        self.backbone = nn.Sequential(
                            nn.Linear(5, 128), 
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
        self.recurrent = nn.GRUCell(4, 4)
    def forward(self, x):
        x = self.backbone(x)
        video_prediction = []
        for i in range(self.video_length):
            x = self.recurrent(get_iteration_noise(x.size(0),self.observation_dim) ,hx)
            video_prediction.append(x)
        return video_prediction



class Discriminator(nn.Module):
    def __init__(self, in_features,attribute_dimensionality):

        super(Discriminator, self).__init__()
        """
         
        :returns
        """
        self.attribute_dimensionality =  attribute_dimensionality
        self.in_features = in_features

        self.backbone = nn.Sequential(
                            nn.Linear(self.in_features, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
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
        self.adv_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128, 50), nn.Softmax())
    def forward(self, x, labels):

        out = self.backbone(x)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

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

env = gym.make(ENV_NAME)
po = print_obs
actor = Actor()
#summary(actor, (4,))
observation_buffer = []
shape_sum = 0
action_loss = torch.nn.CrossEntropyLoss()
state_loss = torch.nn.MSELoss()
if cuda:
    print(f"cuda is {cuda } ")
    action_loss.cuda()
    state_loss.cuda()
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
our_goal = np.array([0,0])
desired_reward = 1
while True:
    if epoch % 10 == 0: 
        print(f"epoch is {epoch } ")
    epoch += 1
    obs_list = env.reset()
    wdx = 0
    total_reward = 0
    while True:
        wdx += 1
#        if wdx % 50 == 0: 
#            print(f"len(my_buffer) is {len(my_buffer) } ")
        obs_list = obs_list[:2]
        if not len(my_buffer) == 50000:  
#            print('Taking Random Action')
            action = env.action_space.sample()
        else: 
            with torch.no_grad():
                cur_obs = FT([obs_list.tolist() + our_goal.tolist() + [desired_reward]] )
                predicted_actions, predicted_state,pred_rew= actor(cur_obs)
                action = torch.argmax(predicted_actions).cpu().numpy()
         
        if len(my_buffer) == 50000 : env.render()
        goal,reward,terminal,_ = env.step(action)    
        goal = goal[:2]
        total_reward += reward
#        if len(my_buffer) == 50000: env.render()
        observation = obs_list.tolist() + goal.tolist() + [reward]
        add_to_buffer(my_buffer, [np.array(observation).astype(np.float32), action, goal, reward] , opt.max_num_items)
           
        
        obs_list = goal
        if terminal: 
            print(f"total_reward is {total_reward } ")
            print(f"terminal is {terminal } ")
            break 
    qt = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    if len(my_buffer) == 50000:
#        print(f"len(my_buffer) is {len(my_buffer) } ")
        for this_batch in range(opt.num_batches):
            batch =  random.sample(my_buffer, opt.batch_size)
            Goals = FT(torch.from_numpy(np.array([sample[0] for sample in batch]).squeeze()).cuda())
            Actions = LT(np.array([sample[1] for sample in batch]))
            GTs = FT(np.array([sample[2] for sample in batch]))
            Rewards = FT(np.array([sample[3] for sample in batch]))
            predicted_actions, pred_state, pred_rew= actor(Goals)
            a1loss = al(predicted_actions,Actions)
            s1loss = state_loss(pred_state, GTs)
            r1loss = state_loss(pred_rew, Rewards)
            print(f"s1loss is {s1loss } ")
            print(f"r1loss is {r1loss } ")
            act_loss_total = a1loss  + s1loss + r1loss
#            print(f"act_loss_total is {act_loss_total } ")
            act_loss_total.backward()
            optimizer.step()
    

    
    
    
#    print('\n\n\n')
#    print(f"new_obs_list is")
#    print_obs(new_obs_list)
#    print('\n\n\n')
#    print(f"new obs is")
#    print_obs(gt)
    
   
     
     
      

print(f"shape_sum is {shape_sum } ")
    

