import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import tqdm
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import Dataset, TensorDataset
import argparse
import os
from Model import ER_MLP, KGMTL
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def make_train_step(model, loss_fn, loss_mse, loss_soft, optimizer):
    # Builds function that performs a step in the train loop
    def train_step_triplet(x, y):  
        output1 = model.StructNet_forward(x[:,0], x[:,1], x[:,2])
        loss_1 = loss_fn(output1, torch.reshape(y, (-1,1)))
        return loss_1
    def AttrNet_h_forward(x, y):   
        output2 = model.head_att_forward(x[:,0], x[:,1])
        loss_2 = loss_mse(output2, torch.reshape(y.float(), (-1,1)))
        return loss_2
    def AttrNet_t_forward(x, y): 
        output3 = model.tail_att_forward(x[:,0], x[:,1])
        loss_3 = loss_mse(output3, torch.reshape(y.float(), (-1,1)))
        return loss_3
    def DescNet_forward(x, y):
        output4 = model.text_forward(x[:,0].long(), x[:,1:301].float())
        loss_4 = loss_soft(output4, y)
        return loss_4
    def param_update(loss):
        loss.backward()
        optimizer.step()
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step_triplet, train_step_head_att, train_step_tail_att, train_step_text_net, param_update


def eval_function(dict_test_positive_user2item, dict_occ_ceid, dict_neg_instances, nb_hist, hit):
    with torch.no_grad():
        hit_10 = 0
        mrr_10 = 0
        K = 0
        r_travel = torch.tensor(idx_travel).to(device)
        #dft_att = torch.tensor(len(dict_att_2_idx)).to(device)
        #print(dft_att)
        pos_score_l=[]
        neg_score_l=[]
        counter_l=[]
        for key in dict_test_positive_user2item:
            if key in dict_occ_ceid and dict_occ_ceid[key] >= nb_hist:
                embd_CEID = torch.tensor(dict_ent_2_idx[key]).to(device)
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    K = K + 1 
                    l_neg = dict_neg_instances[key]
                    for el in l_pos:
                        airport = torch.tensor(dict_ent_2_idx[el]).to(device)
                        pos_score = model.triplet_forward(embd_CEID, r_travel, airport)#, dft_att, dft_att)
                        pos_score = sigmoid(pos_score.detach().cpu().numpy()[0])
                        pos_score_l.append(pos_score)
                        ##
                        counter = 0
                        vec_embed_ceid_ = list()
                        vec_embed_airport_neg_ = list()
                        vec_embed_r_travel_ = list()
                        for el_neg in l_neg:
                            vec_embed_airport_neg_.append(dict_ent_2_idx[el_neg])
                            vec_embed_ceid_.append(dict_ent_2_idx[key])
                            vec_embed_r_travel_.append(idx_travel)
                        vec_embed_airport_neg = torch.tensor(np.array(vec_embed_airport_neg_, dtype=int)).to(device)
                        vec_embed_ceid = torch.tensor(np.array(vec_embed_ceid_, dtype=int)).to(device)
                        vec_embed_r_travel = torch.tensor(np.array(vec_embed_r_travel_, dtype=int)).to(device)   
                        neg_score_list = model.triplet_forward(vec_embed_ceid, vec_embed_r_travel, vec_embed_airport_neg)#, dft_att, dft_att)
                        sig_neg_score_list = sigmoid(neg_score_list.detach().cpu().numpy())
                        for neg_score in sig_neg_score_list:
                            neg_score_l.append(neg_score[0])
                            if pos_score > neg_score[0]:
                                counter = counter + 1
                        counter_l.append(counter)
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter + 1)
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K
    return hit_10, mrr_10, K, pos_score_l, neg_score_l, counter_l

    ### 
    def sigmoid(x):
        return 1/(1 + np.exp(-x))