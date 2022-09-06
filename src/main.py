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
import sys,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append('../KGMTL4Rec')
from Model import ER_MLP, KGMTL
from Evaluation import *
from Data_Processing import *, KGMTL_Data

def main():
    parser = argparse.ArgumentParser(description='KGMTL4REC')

    parser.add_argument('-ds', type=str, required=False, default="Data_MLT")
    parser.add_argument('-epochs', type=int, required=False, default=20)
    parser.add_argument('-batch_size', type=float, required=False, default=524)
    parser.add_argument('-lr', type=float, required=False, default=0.0001)
    parser.add_argument('-model_path', type=str, required=False, default='MLT')
    parser.add_argument('-emb_size', type=int, required=False, default=128)
    parser.add_argument('-hidden_size', type=int, required=False, default=64)
    parser.add_argument('-word_embd_size', type=int, required=False, default=300)
    parser.add_argument('-nb_items', type=int, required=False, default=138)
    parser.add_argument('-Ns', type=int, required=False, default=3)
    parser.add_argument('-device', type=str, required=False, default="cuda:0")
    parser.add_argument('-nb_hist', type=int, required=False, default=1)
    parser.add_argument('-hit', type=int, required=False, default=10)
    args = parser.parse_args()

    ds_path = args.ds
    epochs = args.epochs
    learning_rate = args.lr
    model_path = args.model_path
    emb_size = args.emb_size
    hidden_size = args.hidden_size
    word_embd_size = args.word_embd_size
    nb_items = args.nb_items
    batch_size = args.batch_size
    Ns = args.Ns
    hit = args.hit
    nb_hist = args.nb_hist
    

    ##****** Set Device ******
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 

    device = torch.device(dev)  

    # Now we can create a model and send it at once to the device
    model = KGMTL(len(dict_ent_2_idx), len(dict_rel_2_idx), len(dict_att_2_idx)+1, emb_size, hidden_size, word_embd_size, nb_items, 64, 2, 5)
    model.to(device)
    # We can also inspect its parameters using its state_dict
    print(model)
    
    ## Define losses
    loss_fn = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()
    loss_soft = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Creates the train_step function for our model, loss function and optimizer
    train_step_triplet, train_step_head_att, train_step_tail_att, train_step_text_net, param_update = make_train_step(model, loss_fn, loss_mse, loss_soft, optimizer)

    ## Load Data
    KGMTL_Data = KGMTL_Data(ds_path, Ns)
    
    X_train_triples, y_train_triplets = create_triplets_data()
    X_train_head_attr, X_train_tail_attr, y_train_head_attr, y_train_tail_attr = create_attr_net_data()
    
    X_val_neg, y_val_neg = create_val_data()
    
    dict_val_positive_user2item, dict_neg_instances, dict_occ_ceid = eval_data()
    
    train_loader_triplets, train_loader_head_attr, train_loader_tail_attr, train_loader_desc = create_pytorch_data(X_train_triplets, y_train_triplets, X_train_head_attr, y_train_head_attr, X_train_tail_attr, y_train_tail_attr, X_train_desc_vec, y_train_desc_vec,
                           batch_size)
    
    ## Training the model
    hits_10 = []
    mrrs_10 = []
    tr_loss = []
    val_loss = []
    for epoch in tqdm.tqdm(range(epochs)):
        loss_1_epoch = []; loss_2_epoch = []; loss_3_epoch = []; loss_4_epoch=[]
        for x_batch_triplets, y_batch_triplets in train_loader_triplets:
            optimizer.zero_grad()
            loss1 = train_step_triplet(x_batch_triplets.to(device), y_batch_triplets.to(device))
            loss_1 = param_update(loss1)
            loss_1_epoch.append(loss_1)
            ##
        print('epoch {}, Struct Training loss {}'.format(epoch, np.mean(loss_1_epoch)))
        for x_batch_head_attr, y_batch_head_attr in train_loader_head_attr:
            optimizer.zero_grad()
            loss2 = train_step_head_att(x_batch_head_attr.to(device), y_batch_head_attr.to(device))
            loss_2 = param_update(loss2)
            loss_2_epoch.append(loss_2)
        ##
        print('epoch {}, Head Reg Training loss {}'.format(epoch, np.mean(loss_2_epoch)))
        for x_batch_tail_attr, y_batch_tail_attr in train_loader_tail_attr:
            optimizer.zero_grad()
            loss3 = train_step_tail_att(x_batch_tail_attr.to(device), y_batch_tail_attr.to(device))
            loss_3 = param_update(loss3)
            loss_3_epoch.append(loss_3)
        print('epoch {}, Tail Reg Training loss {}'.format(epoch, np.mean(loss_3_epoch)))
        for x_batch_desc_attr, y_batch_desc_attr in train_loader_desc: 
            optimizer.zero_grad()
            loss4 = train_step_text_net(x_batch_desc_attr.to(device), y_batch_desc_attr.to(device))
            loss_4 = param_update(loss4)
            loss_4_epoch.append(loss_4)
        print('epoch {}, Desc Softmax Training loss {}'.format(epoch, np.mean(loss_4_epoch)))
        ## Total loss
        print('epoch {}, SUM Training loss {}'.format(epoch, np.mean(loss_1_epoch) +  np.mean(loss_2_epoch) + np.mean(loss_3_epoch)))
        tr_loss.append(np.mean(loss_1_epoch) +  np.mean(loss_2_epoch) + np.mean(loss_3_epoch) + np.mean(loss_4_epoch))
        model.eval()
        with torch.no_grad():
            outputs_valid = model.triplet_forward(x_valid_tensor[:, 0], x_valid_tensor[:, 1], x_valid_tensor[:, 2])
            loss_valid = loss_fn(outputs_valid, torch.reshape(y_valid_tensor, (-1,1)))
            print('epoch {}, Validation loss {}'.format(epoch, loss_valid))
            val_loss.append(loss_valid)
            hit_10, mrr_10, K, pos_score_l, neg_score_l, counter_l = eval_function(dict_test_positive_user2item, dict_occ_ceid, dict_neg_instances, nb_hist, hit)
            print('Hit@10: ', hit_10)
            print('MRR@10: ', mrr_10)
            hits_10.append(hit_10)
            mrrs_10.append(mrr_10)
        model.train()
    
if __name__ == '__main__':
    main()    
