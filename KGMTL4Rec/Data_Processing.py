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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from Model import ER_MLP, KGMTL
from Evaluation import *


class KGMTL_Data():
    
    def __init__(self, ds_path, Ns):
        
        ## Number of neg samples
        self.Ns = Ns
        ## Load Data
        self.train_data = pd.read_csv(ds_path + 'CEM-num-std-MTL-train.txt', sep='\t', names=['s', 'p', 'o'])
        self.val_data = pd.read_csv(ds_path + 'CEM-num-std-MTL-val.txt', sep='\t', names=['s', 'p', 'o'])
        ## Group Attributes
        self.attributes = ['<http://data.amadeus.com/ontology/advancedPurchase>','<https://schema.org/totalPrice>',
        '<http://data.amadeus.com/ontology/numberInParty>','<https://schema.org/flightDistance>',
        '<http://data.amadeus.com/ontology/stayDuration>','<http://www.wikidata.org/prop/direct/P2046>',
        '<http://www.wikidata.org/prop/direct/P1081>','<http://www.wikidata.org/prop/direct/P1082>',
        '<http://www.wikidata.org/prop/direct/P2131>','<http://www.wikidata.org/prop/direct/P2044>']

        ## 
        self.train_data_attr = train_data[train_data.p.isin(attributes)]
        ## 
        self.train_data_no_attr = train_data[~train_data.p.isin(attributes)]

        ## Group Entitites
        self.entities_h = set(list(self.train_data_attr.s.value_counts().index))
        self.entities_t = set(list(self.train_data_no_attr.o.value_counts().index))
        self.entities = self.entities_h.union(self.entities_t)

        ## Group Relations
        self.relations = self.train_data_no_attr.p.value_counts().index

        ## Dict Entites and relations
        self.dict_ent_2_idx = dict(zip(self.entities, np.arange(0, len(self.entities), 1)))
        self.dict_rel_2_idx = dict(zip(self.relations, np.arange(0, len(self.relations), 1)))
        self.dict_att_2_idx = dict(zip(self.attributes, np.arange(0, len(self.attributes), 1)))

    def create_triplets_data():
        
        ## Dict contains all Graph objects
        dict_all_2_idx = {}
        dict_all_2_idx.update(self.dict_ent_2_idx)
        dict_all_2_idx.update(self.dict_rel_2_idx)
        dict_all_2_idx.update(self.dict_att_2_idx)
        
        ## Construct positive instances TravelTo property
        train_travel = self.train_data[self.train_data.p=='<http://data.amadeus.com/ontology/travelTo>']
        X_train_pos = np.empty([train_travel.shape[0], train_travel.shape[1]], dtype=int)
        for i, el in enumerate(train_travel.values):
            X_train_pos[i] = [dict_all_2_idx[el_] for el_ in el]
        y_train_pos = np.ones((X_train_pos.shape[0],1))

        ## Construct negative instances (Ns = 3)
        ## idx of travel to relation
        idx_rel_travel_to = self.dict_rel_2_idx['<http://data.amadeus.com/ontology/travelTo>']
        list_dest = np.unique(X_train_pos[:,2])
        ## Create dict of train instances
        dict_train_positive_user2item = dict()
        for el in X_train_pos:
            if el[1] == idx_rel_travel_to:
                CEURI = el[0]
                if not(CEURI in dict_train_positive_user2item):
                    dict_train_positive_user2item[CEURI] = [el[2]]
                else:
                    l = dict_train_positive_user2item[CEURI]
                    l.append(el[2])
                    dict_train_positive_user2item[CEURI] = l
        ## Create the neg instance
        dict_neg_instances = dict()
        for key in dict_train_positive_user2item:
            l=list()
            for i in range(1000):
                el = random.choice(list_dest)
                if not(el in dict_train_positive_user2item[key]):
                    l.append(el)
                if len(l)==self.Ns:
                    break
            dict_neg_instances[key] = l

        ## Create X_train neg sample
        X_train_neg = np.empty([len(dict_neg_instances)*Ns, train_data.shape[1]], dtype=int)
        k = 0
        for key in dict_neg_instances:
            X_train_neg[k] = [key, idx_rel_travel_to, dict_neg_instances[key][0]]
            X_train_neg[k+1] = [key, idx_rel_travel_to, dict_neg_instances[key][1]]
            X_train_neg[k+2] = [key, idx_rel_travel_to, dict_neg_instances[key][2]]
            X_train_neg[k+3] = [key, idx_rel_travel_to, dict_neg_instances[key][3]]
            k = k + self.Ns
        y_train_neg = np.zeros((X_train_neg.shape[0],1))
        
        ### All triplets
        train_all = self.train_data_no_attr[train_data_no_attr.p!='<http://data.amadeus.com/ontology/travelTo>']
        X_train_all_pos = np.empty([self.train_all.shape[0], self.train_all.shape[1]], dtype=int)
        for i, el in enumerate(train_all.values):
            X_train_all_pos[i] = [self.dict_all_2_idx[el_] for el_ in el]
        y_train_all_pos = np.ones((X_train_all_pos.shape[0],1))
        ## Construct negative instances for all other data (Ns = 1)
        ## idx of travel to relation
        list_all_ent_j = np.unique(X_train_all_pos[:,2])
        ## Create dict of train instances
        dict_train_positive_user2item = dict()
        for el in X_train_all_pos:
            CEURI = el[0]
            if not(CEURI in dict_train_positive_user2item):
                dict_train_positive_user2item[CEURI] = [el[2]]
            else:
                l = dict_train_positive_user2item[CEURI]
                l.append(el[2])
                dict_train_positive_user2item[CEURI] = l
        ## Create the neg instance
        dict_neg_instances = dict()
        for key in dict_train_positive_user2item:
            l=list()
            for i in range(1000):
                el = random.choice(list_all_ent_j)
                if not(el in dict_train_positive_user2item[key]): #and not(el in dict_val_positive_user2item):
                    l.append(el)
                if len(l)==self.Ns:
                    break
            dict_neg_instances[key] = l
        ## Create X_train neg sample
        X_train_all_neg = np.empty([len(dict_neg_instances)*Ns, train_data.shape[1]], dtype=int)
        k = 0
        for key in dict_neg_instances:
            for i in range(self.Ns):
                X_train_all_neg[k+i] = [key, idx_rel_travel_to, dict_neg_instances[key][i]]
            k = k + Ns
        y_train_all_neg = np.zeros((X_train_all_neg.shape[0],1))
        
        ## Concatenate positive and negative instances
        X_train_triplets = np.concatenate((X_train_pos, X_train_neg, X_train_all_pos, X_train_all_neg), axis=0)
        y_train_triplets = np.concatenate((y_train_pos, y_train_neg, y_train_all_pos, y_train_all_neg), axis=0)
        
        return X_train_triplets, y_train_triplets
    
    def val_transform(x):
        if x[1] in quant_list:
            v = int(x[2].split(' ')[0])
        elif x[1] in float_list:
            v = float(x[2].split('^^')[0].strip('"'))
        elif x[1] in int_list:       
            v = int(x[2].split('^^')[0].strip('"'))
        return v

    def create_attr_net_data():
        dict_vals_r = dict()
        for el in self.train_data_attr.values:
            v = val_transform(el)
            r = self.dict_att_2_idx[el[1]]
            if r in dict_vals_r:
                l = dict_vals_r[r]
                l.append(v)
                dict_vals_r[r] = l
            else:
                dict_vals_r[r] = [v]
        dict_scaler = dict()
        for key in dict_vals_r:
            scaler = MinMaxScaler()
            X = np.array(dict_vals_r[key]).reshape((-1,1))
            scaler.fit(X)
            dict_scaler[key] = scaler
            
        dict_e2rv = dict()
        for el in self.train_data_attr.values:
            r = self.dict_att_2_idx[el[1]]
            scaler_r = dict_scaler[r]
            v = scaler_r.transform(np.array([val_transform(el)]).reshape((-1,1)))[0][0]
            e = self.dict_all_2_idx[el[0]]
            if e in dict_e2rv:
                l = dict_e2rv[e]
                l.append([r,v])
                dict_e2rv[e] = l
            else:
                dict_e2rv[e] = [[r,v]]
                
        X_list_head_attr = list()
        y_list_head_attr = list()
        X_list_tail_attr = list()
        y_list_tail_attr = list()
        ##
        for i, triple in enumerate(X_train_triplets):
            ei = triple[0]
            rk = triple[1]
            ej = triple[2]
            if ei in dict_e2rv:
                l_vals = dict_e2rv[ei]
                for el in l_vals:
                    vi = el[1]
                    ai = el[0]
                    X_list_head_attr.append([ei, ai])
                    y_list_head_attr.append([vi])
            if ej in dict_e2rv:
                l_vals = dict_e2rv[ej]
                for el in l_vals:
                    vj = el[1]
                    aj = el[0]
                    X_list_tail_attr.append([ej, aj])
                    y_list_tail_attr.append([vj])
        
        X_train_head_attr = np.array(X_list_head_attr, dtype=int).reshape((len(X_list_head_attr), 2))
        X_train_tail_attr = np.array(X_list_tail_attr, dtype=int).reshape((len(X_list_tail_attr), 2))
        ##
        y_train_head_attr = np.array(y_list_head_attr, dtype=int).reshape((len(X_list_head_attr), 1))
        y_train_tail_attr = np.array(y_list_tail_attr, dtype=int).reshape((len(X_list_tail_attr), 1))
        
        return X_train_head_attr, X_train_tail_attr, y_train_head_attr, y_train_tail_attr
    
    def create_val_data():
        ## Construct positive instances
        val_travel = self.val_data[self.val_data.p=='<http://data.amadeus.com/ontology/travelTo>']
        X_val_pos = np.empty([val_travel.shape[0], val_travel.shape[1]], dtype=int)
        for i, el in enumerate(val_travel.values):
            X_val_pos[i] = [self.dict_all_2_idx[el_] for el_ in el]
        y_val_pos = np.ones((X_val_pos.shape[0],1))
        ###

        ## Construct negative instances (Ns = 3)
        ## idx of travel to relation
        idx_rel_travel_to = self.dict_rel_2_idx['<http://data.amadeus.com/ontology/travelTo>']
        list_dest = np.unique(X_train_pos[:,2])
        ## Create dict of train instances
        dict_train_positive_user2item = dict()
        for el in X_train_pos:
            if el[1] == idx_rel_travel_to:
                CEURI = el[0]
                if not(CEURI in dict_train_positive_user2item):
                    dict_train_positive_user2item[CEURI] = [el[2]]
                else:
                    l = dict_train_positive_user2item[CEURI]
                    l.append(el[2])
                    dict_train_positive_user2item[CEURI] = l
        ## Create dict of val instances
        dict_val_positive_user2item = dict()
        for el in X_val_pos:
            if el[1] == idx_rel_travel_to:
                CEURI = el[0]
                if not(CEURI in dict_val_positive_user2item):
                    dict_val_positive_user2item[CEURI] = [el[2]]
                else:
                    l = dict_val_positive_user2item[CEURI]
                    l.append(el[2])
                    dict_val_positive_user2item[CEURI] = l
        ## Create the neg instance
        dict_neg_instances = dict()
        for key in dict_val_positive_user2item:
            l=list()
            for i in range(100):
                el = random.choice(list_dest)
                if not(el in dict_train_positive_user2item[key]) and not (el in dict_val_positive_user2item[key]): #and not(el in dict_val_positive_user2item):
                    l.append(el)
                if len(l)==self.Ns:
                    break
            dict_neg_instances[key] = l

        ## Create X_train neg sample
        X_val_neg = np.empty([len(dict_neg_instances)*Ns, val_data.shape[1]], dtype=int)
        k = 0
        for key in dict_neg_instances:
            X_val_neg[k] = [key, idx_rel_travel_to, dict_neg_instances[key][0]]
            X_val_neg[k+1] = [key, idx_rel_travel_to, dict_neg_instances[key][1]]
            X_val_neg[k+2] = [key, idx_rel_travel_to, dict_neg_instances[key][2]]
            X_val_neg[k+3] = [key, idx_rel_travel_to, dict_neg_instances[key][3]]
            k = k + self.Ns
        y_val_neg = np.zeros((X_val_neg.shape[0],1))
        
        return X_val_neg, y_val_neg
    
    def create_pytorch_data(X_train_triplets, y_train_triplets, X_train_head_attr, y_train_head_attr,
                           X_train_tail_attr, y_train_tail_attr, X_train_desc_vec, y_train_desc_vec,
                           batch_size):
        # Wait, is this a CPU tensor now? Why? Where is .to(device)?
        x_train_tensor_triplets = torch.from_numpy(X_train_triplets)
        y_train_tensor_triplets = torch.from_numpy(y_train_triplets)
        train_data_triplets = TensorDataset(x_train_tensor_triplets, y_train_tensor_triplets)
        train_loader_triplets = DataLoader(dataset=train_data_triplets, batch_size=batch_size, shuffle=True)
        ##
        x_train_tensor_head_attr = torch.from_numpy(X_train_head_attr)
        y_train_tensor_head_attr = torch.from_numpy(y_train_head_attr)
        train_data_head_attr = TensorDataset(x_train_tensor_head_attr, y_train_tensor_head_attr)
        train_loader_head_attr = DataLoader(dataset=train_data_head_attr, batch_size=batch_size, shuffle=True)
        ##
        x_train_tensor_tail_attr = torch.from_numpy(X_train_tail_attr)
        y_train_tensor_tail_attr = torch.from_numpy(y_train_tail_attr)
        train_data_tail_attr = TensorDataset(x_train_tensor_tail_attr, y_train_tensor_tail_attr)
        train_loader_tail_attr = DataLoader(dataset=train_data_tail_attr, batch_size=batch_size, shuffle=True)
        ##
        ##
        x_train_desc = torch.from_numpy(X_train_desc_vec)
        y_train_desc = torch.from_numpy(y_train_desc_vec)
        train_data_desc = TensorDataset(x_train_desc, y_train_desc)
        train_loader_desc = DataLoader(dataset=train_data_desc, batch_size=batch_size, shuffle=True)

        return train_loader_triplets, train_loader_head_attr, train_loader_tail_attr, train_loader_desc
    
    def eval_data():
        ## Data processing for evaluation
        idx_travel = self.dict_rel_2_idx['<http://data.amadeus.com/ontology/travelTo>']
        ### Load val and train data (CEM-all)
        train_data = self.train_data[self.train_data.p=='<http://data.amadeus.com/ontology/travelTo>']

        ### Compute dictionnary of occurences of travels per ceid
        dict_occ_ceid = dict(zip(list(self.train_data.s.value_counts().index), self.train_data.s.value_counts().values))

        ### Compute dictionnary of positive interactions (for LOO eval protocol)
        dict_val_positive_user2item = dict()
        for el in self.val_data.values:
            if el[0] in dict_val_positive_user2item:
                l = dict_val_positive_user2item[el[0]]
                l.append(el[2])
                dict_val_positive_user2item[el[0]] = l
            else:
                dict_val_positive_user2item[el[0]] = [el[2]]

        ### Compute dictionnary of negative interactions (for LOO protocol)
        dict_train_positive_user2item = dict()
        for el in self.train_data.values:
            CEURI = el[0]
            if not(CEURI in dict_train_positive_user2item):
                dict_train_positive_user2item[CEURI] = [el[2]]
            else:
                l = dict_train_positive_user2item[CEURI]
                l.append(el[2])
                dict_train_positive_user2item[CEURI] = l

        list_destinations = list(self.train_data.o.value_counts().index)

        dict_neg_instances = dict()
        for key in dict_train_positive_user2item:
            l=list()
            for el in list_destinations:
                if key in dict_val_positive_user2item:
                    if not(el in dict_train_positive_user2item[key]) and not(el in dict_val_positive_user2item[key]):
                        l.append(el)
            dict_neg_instances[key] = l
            
    return dict_val_positive_user2item, dict_neg_instances, dict_occ_ceid 