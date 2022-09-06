import numpy as np
import math
import pandas as pd
from ast import literal_eval as make_tuple
from time import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Multiply, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import pickle


class CF_Eval():
    
    def __init__(self, model, K, CF_Data):
        self.model = model
        self.K = K
        self.CF_Data = CF_Data
    
    def evaluate_models():
        hit_10 = 0
        mrr_10 = 0
        train_data_mat = self.CF_Data.df_train_data.values
        user_vec = list()
        item_vec = list()
        user_vec = list()
        item_vec = list()
        for key in tqdm.tqdm(self.CF_Data.dict_test_data):
            ##
            user_id = key
            ##
            user_vec.append(np.array([user_id]*len(self.CF_Data.possible_instances)).reshape((-1,1)))
            item_vec.append(np.array(self.CF_Data.possible_instances).reshape((-1,1)))
            ##
        flat_user_vec = np.array([item for sublist in user_vec for item in sublist]).reshape((-1,1))
        ##
        flat_item_vec = np.array([item for sublist in item_vec for item in sublist]).reshape((-1,1))
        ##
        scores_vec = self.model.predict(x=[flat_user_vec, flat_item_vec], batch_size=10000, verbose=1).flatten()
        j = 0
        for key in tqdm.tqdm(self.CF_Data.dict_test_data):       
            ##
            dest_id = self.CF_Data.dict_test_data[key][0]
            ##
            ranked_list = list(np.argsort(scores_vec[j*len(self.CF_Data.possible_instances):(j+1)*len(self.CF_Data.possible_instances)])[::-1][:len(self.CF_Data.possible_instances)])
            ##
            j = j + 1
            dict_train_list = self.CF_Data.dict_train_data[user_id]
            ##
            rank = 0
            for t_id in ranked_list:
                el = t_id
                if not(el in dict_train_list):
                    rank = rank + 1
                    if el == dest_id or rank > self.K:
                        break
            if rank <= self.K:
                hit_10 = hit_10 + 1
                mrr_10 = mrr_10 + 1/rank
        hit_10 = hit_10/len(CF_data.dict_test_data)
        mrr_10 = mrr_10/len(CF_data.dict_test_data)
        return hit_10, mrr_10

    def train_models(train_features, train_labels, test_features, test_labels, batch_size, epochs, verbose=2):
        tr_losses = []; hits_10=[]
        test_losses = []; mrrs_10=[]
        for e in range(epochs):
            print('****************************')
            print("\nEpoch n°{}/{}".format(e+1, epochs))
            history = self.model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
            tr_loss = history.history["loss"][0]
            print("Training Loss", tr_loss)
            ## Compute test loss
            test_loss = self.model.evaluate(x=test_features, y=test_labels)
            print("Test Loss", test_loss)
            ##
            hit_10, mrr_10, ndcg_10 = evaluate_models(self.model)
            #
            print('hit@10:', round(hit_10*100,2), '%')
            ##
            print('mrr_10@10:', round(mrr_10*100,2), '%')
            ##
            print('ndcg_10@10:', ndcg_10)
            ##
            tr_losses.append(tr_loss)   
            test_losses.append(test_loss)
            hits_10.append(hit_10)  
            mrrs_10.append(mrr_10)
        return np.array(tr_losses), np.array(test_losses), np.array(hits_10), np.array(mrrs_10)
    
class FM_Eval():
    
    def __init__(self, model, K, FM_Data):
        self.model = model
        self.K = K
        self.FM_Data = FM_Data
    
    def evaluate_models(test_interactions, test_labels, test_context_features):
        hit_10 = 0
        mrr_10 = 0
        ndcg_10 = 0
        train_data_mat = self.FM_Data.df_train_data.values
        l=0
        ##
        for j, el in tqdm.tqdm(enumerate(test_interactions)):  
            if test_labels[j] == 1:
                l=l+1
                ##
                user_id = el[0]
                dest_test_id = el[1]
                ##
                user_vec = np.array([user_id]*len(self.FM_Data.possible_instances)).reshape((-1,1))
                item_vec = np.array(self.FM_Data.possible_instances).reshape((-1,1))
                X_context = np.zeros((len(self.FM_Data.possible_instances), test_context_features.shape[1]))
                for a in range(len(self.FM_Data.possible_instances)):
                    X_context[a,:] = test_context_features[j,:]
                ##
                X_context[dest_test_id, :] = test_context_features[j,:]
                X = np.zeros((len(user_vec), (self.FM_Data.n_users+self.FM_Data.n_des)), dtype='uint8')
                for i in range(len(user_vec)):
                    user_index = user_vec[i][0]
                    item_index = item_vec[i][0] + n_users
                    ##
                    X[i, user_index] = 1
                    X[i, item_index] = 1 
                X = np.concatenate([X, X_context], axis=1)
                ##
                scores_vec = self.model.predict(x=X, batch_size=1000, verbose=0).flatten()
                ranked_list = list(np.argsort(scores_vec)[::-1][:len(scores_vec)])
                ##
                dict_train_list = self.FM_Data.dict_train_data[user_id]
                ##
                rank = 0
                for t_id in ranked_list:
                    el = t_id
                    if not(el in self.FM_Data.dict_train_list):
                        rank = rank + 1
                        if el == dest_test_id or rank > self.K:
                            break
                if rank <= self.K:
                    hit_10 = hit_10 + 1
                    mrr_10 = mrr_10 + 1/rank
        hit_10 = hit_10/l
        mrr_10 = mrr_10/l
        return hit_10, mrr_10, ndcg_10

    def train_models(train_features, train_labels, test_features, test_labels, batch_size, epochs, verbose=2):
        tr_losses = []; test_losses = []; hits_10 = []; mrrs_10 = []
        for e in range(epochs):
            print('****************************')
            print("\nEpoch n°{}/{}".format(e+1, epochs))
            history = self.model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
            tr_loss = history.history["loss"][0]
            print("Training Loss", tr_loss)
            ## Compute test loss
            test_loss = self.model.evaluate(x=test_features, y=test_labels)
            print("Test Loss", test_loss)
            ##
            hit_10, mrr_10, ndcg_10 = evaluate_models(self.model, self.K)
            #
            print('hit@10:', round(hit_10*100,2), '%')
            ##
            print('mrr_10@10:', round(mrr_10*100,2), '%')
            ##
            print('ndcg_10@10:', ndcg_10)
            ##
            tr_losses.append(tr_loss)   
            test_losses.append(test_loss)
            hits_10.append(hit_10)  
            mrrs_10.append(mrr_10)
        return np.array(tr_losses), np.array(test_losses), np.array(hits_10), np.array(mrrs_10)