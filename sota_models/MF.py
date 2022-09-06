import numpy as np
from scipy import sparse
import implicit
import pandas as pd
import tqdm
import argparse

# Load the args parameters
parser = argparse.ArgumentParser()

parser.add_argument('-mn', type=str, required=False, default="BPRMF")
parser.add_argument('-ds', type=str, required=False, default="CEM-travel")
parser.add_argument('-factors_size', type=int, required=False, default=15)
parser.add_argument('-iterations', type=int, required=False, default=20)
args = parser.parse_args()

model_name = args.mn
ds_path = args.ds
factors_size = args.factors_size
iterations = args.iterations

## Load interaction Data
df_interaction_test = pd.read_csv(ds_path + 'Travel_interaction_test.csv')
df_interaction_train = pd.read_csv(ds_path + 'Travel_interaction_train.csv')
## Load train interaction matrix
interaction_matrix_train = np.load(ds_path + 'Travel_interaction_matrix_train.npy')

## Transform data to be used for evaluation
df_train_data = df_interaction_train.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
df_test_data = df_interaction_test.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
dict_train_data = dict(zip(df_train_data.TID, df_train_data.lists))
dict_test_data = dict(zip(df_test_data.TID, df_test_data.lists))

## Sparce interaction matrix
s_interaction_matrix_train = sparse.csr_matrix(interaction_matrix_train)

## Instantiate the model
if model_name=='ImplicitMF':
    model = implicit.als.AlternatingLeastSquares(factors=factors_size, iterations=iterations, use_gpu=False)
elif model_name='BPRMF':
    model = bpr.BayesianPersonalizedRanking(factors=factors_size, iterations=iterations, use_gpu=False)

## Traing the model
alpha = 15
model.fit((alpha*s_interaction_matrix_train).astype('double'))

## Load factors
dest_vecs = model.item_factors 
user_vecs = model.user_factors

## Evaluation of the algorithm
hit_10 = 0
mrr_10 = 0
train_data_mat = df_train_data.values
for key in tqdm.tqdm(dict_test_data):
    ##
    user_id = key
    talk_test_id = dict_test_data[key][0]
    ##
    scores_vec = np.dot(dest_vecs, user_vecs[user_id])
    ranked_list = list(np.argsort(scores_vec)[::-1][:len(scores_vec)])
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
        
## 
print('Hit@10: ', round(hit_10*100/len(dict_test_data), 2), '%')
print('MRR@10: ', round(mrr_10*100/len(dict_test_data),2), '%')
