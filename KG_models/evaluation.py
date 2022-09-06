import pandas as pd
import numpy as np
import pickle as pkl
from numpy.linalg import norm
from collections import Counter
from KG_Models import *
import argparse

# Load the args parameters
parser = argparse.ArgumentParser()

parser.add_argument('-ds', type=str, required=False, default="CEM-travel")
parser.add_argument('-nb_hist', type=int, required=False, default=5)
parser.add_argument('-hit', type=int, required=False, default=10)
parser.add_argument('-embd_size', type=int, required=False, default=50)
args = parser.parse_args()

ds_path = args.ds
nb_hist = args.nb_hist
hit = args.hit
embd_size = args.embd_size

### Load test and train data (CEM-all)
test_data = pd.read_csv('../dataset/CEM-travel/CEM-travel-test.txt', sep='\t', names=['s', 'p', 'o'])
train_data = pd.read_csv('../dataset/CEM-travel/CEM-travel-train.txt', sep='\t', names=['s', 'p', 'o'])

### Compute dictionnary of occurences of travels per tid
dict_occ_tid = dict(zip(list(train_data.s.value_counts().index), train_data.s.value_counts().values))

### Compute dictionnary of positive interactions (for LOO eval protocol)
dict_test_positive_user2item = dict()
for el in test_data.values:
    if el[0] in dict_test_positive_user2item:
        l = dict_test_positive_user2item[el[0]]
        l.append(el[2])
        dict_test_positive_user2item[el[0]] = l
    else:
        dict_test_positive_user2item[el[0]] = [el[2]]
        
### Compute dictionnary of negative interactions (for LOO protocol)
dict_train_positive_user2item = dict()
for el in train_data.values:
    CEURI = el[0]
    if not(CEURI in dict_train_positive_user2item):
        dict_train_positive_user2item[CEURI] = [el[2]]
    else:
        l = dict_train_positive_user2item[CEURI]
        l.append(el[2])
        dict_train_positive_user2item[CEURI] = l
        
list_destinations = list(train_data.o.value_counts().index)
        
dict_neg_instances = dict()
for key in dict_train_positive_user2item:
    l=list()
    for el in list_destinations:
        if not(el in dict_train_positive_user2item[key]) and not(el in dict_test_positive_user2item):
            l.append(el)
    dict_neg_instances[key] = l


list_models = list()
list_hit = list()
list_mrr = list()
list_K = list()

## TransE eval
transe = TransE(ds_path, embd_size, embd_size)
hit_10, mrr_10, K = transe.eval_function(dict_test_positive_user2item, dict_occ_tid, dict_neg_instances, nb_hist, hit)
print('Number of evaluations: ', str(K), '\n')
print('TransE results : \n Hit@10 = ', str(round(hit_10,6)), '\n MRR@10 = ', str(round(mrr_10,6)))
list_models.append('TransE')
list_K.append(hit)
list_hit.append(hit_10)
list_mrr.append(mrr_10)

## TransH eval
transh = TransH(ds_path, embd_size, embd_size)
hit_10, mrr_10, K = transh.eval_function(dict_test_positive_user2item, dict_occ_tid, dict_neg_instances, nb_hist, hit)
print('Number of evaluations: ', str(K), '\n')
print('TransH results : \n Hit@10 = ', str(round(hit_10,6)), '\n MRR@10 = ', str(round(mrr_10,6)))
list_models.append('TransH')
list_K.append(hit)
list_hit.append(hit_10)
list_mrr.append(mrr_10)

## TransR eval
transr = TransR(ds_path, embd_size, embd_size)
hit_10, mrr_10, K = transr.eval_function(dict_test_positive_user2item, dict_occ_tid, dict_neg_instances, nb_hist, hit)
print('Number of evaluations: ', str(K), '\n')
print('TransR results : \n Hit@10 = ', str(round(hit_10,6)), '\n MRR@10 = ', str(round(mrr_10,6)))
list_models.append('TransR')
list_K.append(hit)
list_hit.append(hit_10)
list_mrr.append(mrr_10)

## SLM eval
slm = SLM(ds_path, embd_size, embd_size)
hit_10, mrr_10, K = slm.eval_function(dict_test_positive_user2item, dict_occ_tid, dict_neg_instances, nb_hist, hit)
print('Number of evaluations: ', str(K), '\n')
print('SLM results : \n Hit@10 = ', str(round(hit_10,6)), '\n MRR@10 = ', str(round(mrr_10,6)))
list_models.append('SLM')
list_K.append(hit)
list_hit.append(hit_10)
list_mrr.append(mrr_10)

## SME eval
sme_bl = SME_BL(ds_path, embd_size, embd_size)
hit_10, mrr_10, K = sme_bl.eval_function(dict_test_positive_user2item, dict_occ_tid, dict_neg_instances, nb_hist, hit)
print('Number of evaluations: ', str(K), '\n')
print('SME_BL results : \n Hit@10 = ', str(round(hit_10,6)), '\n MRR@10 = ', str(round(mrr_10,6)))
list_models.append('SME_BL')
list_K.append(hit)
list_hit.append(hit_10)
list_mrr.append(mrr_10)

## ROTATE eval
rotate = RotatE(ds_path, embd_size, embd_size, 0.8)
hit_10, mrr_10, K = rotate.eval_function(dict_test_positive_user2item, dict_occ_tid, dict_neg_instances, nb_hist, hit+1)
print('Number of evaluations: ', str(K), '\n')
print('ROTATE results : \n Hit@10 = ', str(round(hit_10,6)), '\n MRR@10 = ', str(round(mrr_10,6)))
list_models.append('ROTATE')
list_K.append(hit+1)
list_hit.append(hit_10)
list_mrr.append(mrr_10)