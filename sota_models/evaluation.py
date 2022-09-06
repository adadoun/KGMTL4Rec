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
from sota_Models import *
from Data_processing import *
from Models_Evaluation import *
import argparse

## Load the args parameters
parser = argparse.ArgumentParser()

##
parser.add_argument('-mn', type=str, required=True, default="GMF, NCF, DKFM")
parser.add_argument('-ds', type=str, required=True, default="TDNA-travel")
parser.add_argument('-Ns', type=int, required=True, default=3)
parser.add_argument('-batch_size', type=int, required=True, default=256)
parser.add_argument('-epochs', type=int, required=True, default=3)
parser.add_argument('-lr', type=int, required=True, default=0.00001)
parser.add_argument('-embd_size', type=int, required=True, default=128)
parser.add_argument('-K', type=int, required=True, default=10)
args = parser.parse_args()

## 
ds_path = args.ds
Ns = args.Ns
batch_size = args.batch_size;
epochs = args.epochs;
lr = args.lr
embd_size = args.embd_size
K = args.K
mn = args.K

models = mn.split(';')

print('************* Load CF Data ****************')
## Instantiate Interaction Dataset
CF_Data = CF_Data(ds_path, Ns)
## Create val and train data
train_set = CF_Data.create_train_set()
##
test_set = CF_Data.create_test_set()
##
train_set_features = [train_set[:,0], train_set[:,1]]
train_set_labels = train_set[:,2]
##
test_set_features = [test_set[:,0], test_set[:,1]]
test_set_labels = test_set[:,2]


if 'GMF' in models:
    print('*********** Evaluating GMF *************')
    print('\n')
    print('Generalized Matrix Factorization')
    GMF = get_model_GMF(num_users=CF_Data.n_users, num_items=CF_Data.n_talks, lr=lr)
    GMF_Eval = CF_Eval(GMF, K, CF_Data)
    tr_losses, test_losses, hits_10, mrrs_10 = GMF_Eval.train_models(model=GMF, train_features=train_set_features, train_labels=train_set_labels, 
                                       test_features=test_set_features, test_labels=test_set_labels,
                                       batch_size=batch_size, epochs=epochs, K=CF_Eval.K)
    with open('Metrics/hits_GMF.pkl', 'wb') as f:
        pickle.dump(hits_10, f)
    ##
    with open('Metrics/mrrs_GMF.pkl', 'wb') as f:
        pickle.dump(mrrs_10, f)
    
if 'MLP' in models:
    print('\n')
    print('*********** Evaluating MLP *************')
    print('\n')
    print('Multi Layer Perceptron')
    MLP = get_model_MLP(num_users=CF_Data.n_users, num_items=CF_Data.n_talks, lr=lr)
    MLP_Eval = CF_Eval(MLP, K, CF_Data)
    tr_losses, test_losses, hits_10, mrrs_10 = MLP_Eval.train_models(model=MLP, train_features=train_set_features, train_labels=train_set_labels, 
                                       test_features=test_set_features, test_labels=test_set_labels,
                                       batch_size=batch_size, epochs=epochs, K=CF_Eval.K)
    ##
    with open('Metrics/hits_MLP.pkl', 'wb') as f:
        pickle.dump(hits_10, f)
    ##
    with open('Metrics/mrrs_MLP.pkl', 'wb') as f:
        pickle.dump(mrrs_10, f)

if 'NCF' in models:
    print('\n')
    print('*********** Evaluating NCF *************')
    print('\n')
    print('Neural Collaborative Filtering')
    NCF = get_model_NCF(num_users=CF_Data.n_users, num_items=CF_Data.n_talks, lr=lr)
    NCF_Eval = CF_Eval(MLP, K, CF_Data)
    tr_losses, test_losses, hits_10, mrrs_10 = NCF_Eval.train_models(model=NCF, train_features=train_set_features, train_labels=train_set_labels, 
                                       test_features=test_set_features, test_labels=test_set_labels,
                                       batch_size=batch_size, epochs=epochs, K=CF_Eval.K)
    ##
    with open('Metrics/hits_NCF.pkl', 'wb') as f:
        pickle.dump(hits_10, f)
    ##
    with open('Metrics/mrrs_NCF.pkl', 'wb') as f:
        pickle.dump(mrrs_10, f)

if 'FM' in models:
    print('\n')
    print('************* Load FM Data ****************')
    ## Instantiate FM Dataset
    FM_Data = FM_Data(ds_path, Ns)
    ## Create val and train data
    train_set = FM_Data.create_train_set()
    ##
    test_set = FM_Data.create_test_set()
    ##
    tr_interactions, tr_context_features, tr_labels = FM_Data.features_fm_model(train_set)
    test_interactions, test_context_features, test_labels = FM_Data.features_fm_model(test_set)

    ## 
    X_tr_fm, y_tr_fm = FM_Data.compute_fm_features(tr_interactions, tr_context_features, tr_labels)
    X_test_fm, y_test_fm = FM_Data.compute_fm_features(test_interactions, test_context_features, test_labels)

    ##
    print('\n')
    print('*********** Evaluating FM *************')
    print('\n')
    print('Factorization Machine')
    FM = get_model_FM(nb_num_features=FM_Data.n_features, hidden_dim=self.embd_size, lr = lr)
    FM_Eval = FM_Eval(FM, K, CF_Data)
    tr_losses, test_losses, hits_10, mrrs_10 = FM_Eval.train_models(model=FM, train_features=X_tr_fm, train_labels=y_tr_fm, 
                                       test_features=X_test_fm, test_labels=y_test_fm,
                                       batch_size=batch_size, epochs=epochs, K=CF_Eval.K)
    with open('Metrics/hits_FM.pkl', 'wb') as f:
        pickle.dump(hits_10, f)
    ##
    with open('Metrics/mrrs_FM.pkl', 'wb') as f:
        pickle.dump(mrrs_10, f)
    print('\n')

if 'DKFM' in models:
    print('\n')
    print('************* Load DKFM Data ****************)
    ## Instantiate FM Dataset
    DKFM_Data = DKFM_Data(ds_path, Ns)
    ## Create val and train data
    train_set = DKFM_Data.create_train_set()
    ##
    test_set = DKFM_Data.create_test_set()
    ##
    tr_interactions, tr_context_features, tr_features_embed, tr_features_one_hot, tr_num_features, tr_labels = DKFM_Data.features_dkfm_model(df_dkfm_train)
    ##
    test_interactions, test_context_features, test_features_embed, test_features_one_hot, test_num_features, test_labels = DKFM_Data.features_dkfm_model(df_dkfm_test)

    ##
    tr_kge, tr_te = DKFM_Data.compute_embeddings(tr_interactions)
    ##
    test_kge, test_te = DKFM_Data.compute_embeddings(test_interactions)

    n_users = len(np.unique(tr_interactions[:,0]))
    n_des = len(np.unique(tr_interactions[:,1]))
    ##
    kge_dim = 50
    te_dim = 300
    ##
    nb_features_one_hot = tr_features_one_hot.shape[1]
    nb_num_features = tr_num_features.shape[1]
    nb_features_embed = tr_features_embed.shape[1]
    nb_context_features = tr_context_features.shape[1]
    ##
    nb_dpt_days = len(np.unique(tr_features_embed[:, 0]))
    nb_nat = len(np.unique(tr_features_embed[:, 1]))
    nb_iss = len(np.unique(tr_features_embed[:, 2]))
    nb_bkg = len(np.unique(tr_features_embed[:, 3]))

    print('\n')
    print('************ Evaluating DKFM ***************')
    print('DKFM Model')
    print('\n')
    DKFM = get_model_DKFM(num_users=DKFM_Data.n_users, num_items=DKFM_Data.n_des, nb_num_features=nb_num_features, nb_features_one_hot=nb_features_one_hot, nb_context_features=nb_context_features,
                        kge_dim=kge_dim, te_dim=te_dim,
                        nb_dpt_days=nb_dpt_days, nb_nat=nb_nat, nb_iss=nb_iss, nb_bkg=nb_bkg,lr=lr)
    DKFM_Eval = DKFM_Eval(DKFM, K, DKFM_Data)
    train_features = [tr_interactions[:,0], tr_interactions[:,1], tr_features_embed[:,0], tr_features_embed[:,1], tr_features_embed[:,2], tr_features_embed[:,3],
                      tr_features_one_hot, tr_num_features, tr_context_features, tr_kge, tr_te]
    test_features = [test_interactions[:,0], test_interactions[:,1], test_features_embed[:,0], test_features_embed[:,1], test_features_embed[:,2], test_features_embed[:,3],
                      test_features_one_hot, test_num_features, test_context_features, test_kge, test_te]
    tr_losses, test_losses, hits_10, mrrs_10 = DKFM_Eval.train_models(model=DKFM, train_features=train_features, train_labels=tr_labels, 
                                       test_features=test_features, test_labels=test_labels,
                                       batch_size=batch_size, epochs=epochs, K=DKFM_Eval.K)
    with open('Metrics/hits_DKFM.pkl', 'wb') as f:
        pickle.dump(hits_10, f)
    ##
    with open('Metrics/mrrs_DKFM.pkl', 'wb') as f:
        pickle.dump(mrrs_10, f)
    

if 'WDL' in models:
    print('\n')
    print('************* Load WDL Data ****************')
    ## Instantiate FM Dataset
    WDL_Data = WDL_Data(ds_path, Ns)
    ## Create val and train data
    train_set = WDL_Data.create_train_set()
    ##
    test_set = WDL_Data.create_test_set()
    ##
    tr_interactions, tr_features_embed, tr_features_one_hot, tr_num_features, tr_labels = features_wdl_model(train_set)
    test_interactions, test_features_embed, test_features_one_hot, test_num_features, test_labels = features_wdl_model(test_set)
    ##
    nb_features_one_hot = tr_features_one_hot.shape[1]
    nb_num_features = tr_num_features.shape[1]
    nb_features_embed = tr_features_embed.shape[1]
    ##
    nb_dpt_days = len(np.unique(tr_features_embed[:, 0]))
    nb_nat = len(np.unique(tr_features_embed[:, 1]))
    nb_iss = len(np.unique(tr_features_embed[:, 2]))
    nb_bkg = len(np.unique(tr_features_embed[:, 3]))
          
    print('\n')
    print('************ Evaluating WDL ***************')
    print('Wide & Deep Learning Model')
    WDL = get_model_WDL(num_users=n_users, num_items=n_talks, nb_num_features=nb_num_features, nb_features_one_hot=nb_features_one_hot,
                        nb_dpt_days=nb_dpt_days, nb_nat=nb_nat, nb_iss=nb_iss, nb_bkg=nb_bkg, hidden_dim_user=128, hidden_dim_item=64, lr=lr)
    WDL_Eval = WDL_Eval(FM, K, CF_Data)
    train_features = [tr_interactions[:,0], tr_interactions[:,1], tr_features_embed[:,0], tr_features_embed[:,1], tr_features_embed[:,2], tr_features_embed[:,3],
                      tr_features_one_hot, tr_num_features]
    test_features = [test_interactions[:,0], test_interactions[:,1], test_features_embed[:,0], test_features_embed[:,1], test_features_embed[:,2], test_features_embed[:,3],
                      test_features_one_hot, test_num_features]
    tr_losses, test_losses, hits_10, mrrs_10 = train_models(model=WDL, train_features=train_features, train_labels=tr_labels, 
                                       test_features=test_features, test_labels=test_labels,
                                       batch_size=batch_size, epochs=epochs, K=WDL.K)
    with open('Metrics/hits_WDL.pkl', 'wb') as f:
        pickle.dump(hits_10, f)
    ##
    with open('Metrics/mrrs_WDL.pkl', 'wb') as f:
        pickle.dump(mrrs_10, f)
    print('\n')