import numpy as np
from scipy import sparse
import implicit
import pandas as pd
import tqdm

## Load interaction Data
df_interaction_test = pd.read_csv('Data/Travel_interaction_test.csv')
df_interaction_train = pd.read_csv('Data/Travel_interaction_train.csv')
## Load train interaction matrix
interaction_matrix_train = np.load('Data/Travel_interaction_matrix_train.npy')

## Transform data to be used for evaluation
df_train_data = df_interaction_train.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
df_test_data = df_interaction_test.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
dict_train_data = dict(zip(df_train_data.TID, df_train_data.lists))
dict_test_data = dict(zip(df_test_data.TID, df_test_data.lists))

## Most Pop return the top most 10
ranked_list= list(np.arange(0, 134, 1))

## Evaluationof most pop algorithm
hit_10 = 0
mrr_10 = 0
train_data_mat = df_train_data.values
for key in tqdm.tqdm(dict_test_data):
    ##
    user_id = key
    talk_test_id = dict_test_data[key][0]
    ##
    dict_train_list = dict_train_data[user_id]
    ##
    rank = 0
    for el in ranked_list:
        if not(el in dict_train_list):
            rank = rank + 1
            if el == talk_test_id or rank > 10:
                break
    #break
    if rank <= 10:
        hit_10 = hit_10 + 1
        mrr_10 = mrr_10 + 1/rank
        
## Display HR and MRR scores
print('Hit@10: ', round(hit_10*100/len(dict_test_data), 2), '%')
print('MRR@10: ', round(mrr_10*100/len(dict_test_data),2), '%')