import numpy as np

from torch.nn import functional

from sklearn.preprocessing import KBinsDiscretizer

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

ENV_NAME = "Walker2d-v2"
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--num_batches", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_actions", type=int, default=2, help="number of attributes for dataset")
parser.add_argument("--max_num_items", type=int, default=50000, help="number of attributes for dataset")
parser.add_argument("--path_length", type=int, default=50, help="number of steps in path for dataset")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval betwen image samples")

parser.add_argument('--gen_ckpt', help='generator checkpoint')
parser.add_argument('--disc_ckpt', help='discriminator checkpoint')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
class Path_Generator2(nn.Module):
    def __init__(self, path_length, observation_dim):
        """
         
        :returns
        """
        super(Path_Generator, self).__init__()

        self.path_length =  path_length
        self.observation_dim = observation_dim

        self.backbone = nn.Sequential(
                            nn.Linear(9, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
                            nn.Linear(128, 128), 
                            nn.LeakyReLU(),
#                            nn.Linear(128, self.path_length * self.observation_dim), 
#                            nn.LeakyReLU(),
                         )

#        self.emb = nn.Embedding(50, 64)
#        self.recurrent = nn.GRUCell(4, 4)
#        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128, 2), nn.Softmax(dim = 1))
        self.path_layer = nn.Sequential(nn.Linear(128, self.path_length * self.observation_dim))
        self.act_layer = nn.Sequential(nn.Linear(128, 2), nn.Softmax(dim = 1))
    def forward(self, x):
        out = self.backbone(x)
        reward_label = self.aux_layer(out)
        path = self.path_layer(out)
        action = self.act_layer(out)
        path = path.view(path.size(0), self.path_length, self.observation_dim)
        return action, path, reward_label
class Path_Generator(nn.Module):
    def __init__(self, input_features, output_values, path_length, path_feats=5):
        """
         
        :returns
        """
        super(Path_Generator, self).__init__()

        self.path_length =  path_length
#        self.observation_dim = observation_dim

        self.fc1 = nn.Linear(in_features=input_features, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Sequential(nn.Linear(in_features=50 * path_feats, out_features=output_values), nn.Softmax())
        self.fc4 = nn.Linear(in_features=32, out_features=50 * path_feats)
        self.aux_layer = nn.Sequential(nn.Linear(32, 2), nn.Softmax(dim = 1))
    def forward(self, x):
        x = functional.selu(self.fc1(x))
        out = functional.selu(self.fc2(x))
        pl = self.fc4(out)
        pl = pl.view(pl.size(0), -1)
        x = self.fc3(pl)
        reward_label = self.aux_layer(out)
        return x, pl, reward_label




#        hx = self.get_gru_initial_state(num_samples)
#        for i in range(self.path_length):
#            hx = self.recurrent(get_iteration_noise(x.size(0),self.observation_dim) ,hx)
#            video_prediction.append(hx)
#        video_prediction = torch.concat(video_prediction)

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
def optimize_states(model,current_state,path, rewards, actions): 
     
    """
     
    :returns
    """

    optimizer = torch.optim.Adam(params=model.parameters(), lr=.01)
    rewards = np.expand_dims(np.array(rewards), 1)
    current_state = torch.concat([FloatTensor(current_state), FloatTensor(rewards)], 1)
    model.train()
    pred_actions,pred_path, pred_rew = model(FloatTensor(current_state))
    loss = state_loss(path.view(pred_path.size(0),-1), pred_path)
    a1loss = al(pred_actions,LT(actions))
    a2loss = al(pred_rew,LT(rewards.squeeze()))
    our_loss = loss + a1loss + .2* a2loss
    optimizer.zero_grad()
    total_loss = loss.item()
    our_loss.backward()
    optimizer.step()
    model.eval()

    return total_loss
env = gym.make(ENV_NAME)
po = print_obs
path_gen = Path_Generator(5,opt.n_actions, opt.path_length, 5)
#summary(actor, (4,))
observation_buffer = []
shape_sum = 0
action_loss = torch.nn.CrossEntropyLoss()
state_loss = torch.nn.MSELoss()
if cuda:
    print(f"cuda is {cuda } ")
    action_loss.cuda()
    state_loss.cuda()
    path_gen.cuda()
al = action_loss
optimizer = torch.optim.Adam(path_gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
LT = LongTensor
FT = FloatTensor

#print_obs(obs_list)
my_buffer = []
#for epoch in range(opt.n_epochs):
epoch = 0
our_goal = np.array([0,0,0,0])
while True:
    if epoch % 10 == 0: 
        print(f"epoch is {epoch } ")
    epoch += 1
    first_obs_list = env.reset()
#    first_obs_list = np.random.uniform(low=-.12, high = .12, size = (4,))
#    env.state = first_obs_list
    obs_list = first_obs_list
    wdx = 0
    total_reward = 0
    path = []
    actions = []
    states = []
    rewards = []
    max_reward = 10
    desired_reward = 500
    while True:
        wdx += 1
#        if wdx % 50 == 0: 
#            print(f"len(my_buffer) is {len(my_buffer) } ")
#        if  epoch % 10 != 0: #len(my_buffer) == opt.max_num_items:  
        if  True: #len(my_buffer) == opt.max_num_items:  
#            print('Taking Random Action')
            action = env.action_space.sample()
        else: 
            with torch.no_grad():
#                cur_obs = FT([obs_list.tolist() + our_goal.tolist() + [desired_reward]] )
                cur_obs = FT([np.concatenate([obs_list,np.array([1])])])
                predicted_actions, predicted_state,pred_rew= path_gen(cur_obs)
                action = torch.argmax(predicted_actions.squeeze()).cpu().numpy()
                print(f"action is {action } ")
        if wdx ==1: first_action = action 
#        print(f"predicted_actions is {predicted_actions } ")
#        if len(my_buffer) == opt.max_num_items : env.render()
        env.render()
        next_state,reward,terminal,_ = env.step(action)    
#        path.append(np.concatenate([next_state,np.array([reward ])]))
#        states.append(obs_list)
#        actions.append(int(action))
        total_reward += reward
#        rewards.append(reward)
#        if len(my_buffer) == 50000: env.render()
           
        
        obs_list = next_state
        if terminal: 
            
            if epoch % 10 == 0: 
                print(f"total_reward is {total_reward } ")
                print(f"terminal is {terminal } ")
            break 
#    newpaths = [path[i:] for i in range(len(path))]
##    trace()
##    rewards = [sum(rewards[:]) - sum(rewards[i:]) for i in range(len(path))]
##    trace()
#    pathsplits = []
#    for path in newpaths: 
#        if len(path) < 50: 
#            path = path + [path[-1] for i in range(50 - len(path))]
#        path_split = np.array_split(path, 50)
#        path = [random.sample(x.tolist(),1) for x in path_split]
#        path = np.stack(path).squeeze()
#        pathsplits.append(path)
#    pathsplits = np.array(pathsplits)
##    print(f"first_obs_list is {first_obs_list } ")
#    optimize_states(path_gen,states, FloatTensor(pathsplits), rewards, actions)
    
   
  
 


#    observation = first_obs_list.tolist() + next_state.tolist() + [total_reward]
#    add_to_buffer(my_buffer, [np.array(observation).astype(np.float32), first_action, path, total_reward] , opt.max_num_items)
#    if len(my_buffer) == opt.max_num_items:
##        print(f"len(my_buffer) is {len(my_buffer) } ")
#        for this_batch in range(opt.num_batches):
#            batch =  random.sample(my_buffer, opt.batch_size)
#            Goals = FT(torch.from_numpy(np.array([sample[0] for sample in batch]).squeeze()).cuda())
#            Actions = LT(np.array([sample[1] for sample in batch]))
#            GTs = FT(np.array([sample[2] for sample in batch]))
#            Rewards = np.array([[sample[3]] for sample in batch]).astype(np.float32)
#            if this_batch == 0:
#                est = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')
#                est.fit(Rewards)
#            Rewards_binned = est.transform(Rewards)
#            predicted_actions, pred_paths, pred_rew= path_gen(Goals)
##            print(f"pred_rew is {np.argmax(pred_rew.detach().cpu().numpy(), 1) } ")
#            print(f"Goals is {Goals[0] } ")
#            print(f"GTs is {GTs[0] } ")
#            a1loss = al(predicted_actions,Actions)
#            s1loss = state_loss(pred_paths, GTs)
#            r1loss = al(pred_rew, LT(Rewards_binned.squeeze()))
#            print(f"s1loss is {s1loss } ")
#            print(f"r1loss is {r1loss } ")
#            act_loss_total = a1loss  + s1loss + r1loss
##            print(f"act_loss_total is {act_loss_total } ")
#            act_loss_total.backward()
#            optimizer.step()
#    
#
#    
#    
#    
##    print('\n\n\n')
##    print(f"new_obs_list is")
##    print_obs(new_obs_list)
##    print('\n\n\n')
##    print(f"new obs is")
##    print_obs(gt)
#    
#   
#     
#     
#      
#
#print(f"shape_sum is {shape_sum } ")
#    
#
