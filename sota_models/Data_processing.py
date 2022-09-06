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

class CF_Data():

    def __init__(self, ds_path, Ns):  
        ##
        self.Ns = Ns
        ## 
        df_interaction_test = pd.read_csv(ds_path + 'Travel_interaction_test.csv')
        df_interaction_train = pd.read_csv(ds_path + 'Travel_interaction_train.csv')
        self.possible_instances = np.sort(list(df_interaction_train.DEST.value_counts().index))
        ## 
        self.n_talks = len(df_interaction_train.DEST.value_counts())
        self.n_users = len(df_interaction_train.TID.value_counts())
        ##
        df_train_data = df_interaction_train.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        df_test_data = df_interaction_test.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        self.dict_train_data = dict(zip(df_train_data.TID, df_train_data.lists))
        self.dict_test_data = dict(zip(df_test_data.TID, df_test_data.lists))
        
    def create_train_set(self):
        train_set = []
        for key in self.dict_train_data:
            pos_intances = self.dict_train_data[key]
            set_neg_instances = set(self.possible_instances) - set(pos_intances)
            for pos_el in pos_intances:
                train_set.append([key, pos_el, 1])
                for i in range(self.Ns):
                    neg_el = random.choice(list(set_neg_instances))
                    train_set.append([key, neg_el, 0])
        return np.array(train_set)
    
    def create_test_set(self):
        test_set = []
        for key in self.dict_test_data:
            pos_intances = self.dict_test_data[key]
            tr_pos_intances = self.dict_train_data[key]
            set_neg_instances = set(self.possible_instances) - set(pos_intances) - set(tr_pos_intances)
            pos_el = pos_intances[0]
            test_set.append([key, pos_el, 1])
            for i in range(self.Ns):
                neg_el = random.choice(list(set_neg_instances))
                test_set.append([key, neg_el, 0])
        return np.array(test_set)
    
class FM_Data():

    def __init__(self, ds_path, Ns):  
        ## 
        self.Ns = Ns
        ## 
        self.df_data_test = pd.read_csv(ds_path + 'Travel_data_test.csv')
        self.df_data_train = pd.read_csv(ds_path + 'Travel_data_train.csv')
        self.possible_instances = np.sort(list(df_interaction_train.DEST.value_counts().index))
        ## 
        self.n_talks = len(df_interaction_train.DEST.value_counts())
        self.n_users = len(df_interaction_train.TID.value_counts())
        ## 
        df_train_data = df_data_train.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        df_test_data = df_data_test.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        self.dict_train_data = dict(zip(df_train_data.TID, df_train_data.lists))
        self.dict_test_data = dict(zip(df_test_data.TID, df_test_data.lists))
        
    def create_train_set(self):
        mtx = self.df_data_train.values
        lst=[]
        possible_instances = list(self.df_data_train.DESTINATION.value_counts().index)
        l_features = list(self.df_data_train.columns)
        l_features.append('Y')
        for el in mtx:
            pos_intances = self.dict_train_data[el[0]]
            set_neg_instances = set(self.possible_instances) - set(pos_intances)
            pos_set = [el_ for el_ in el]
            pos_set.append(1)
            lst.append(pos_set)
            for i in range(self.Ns):
                neg_el = random.choice(list(set_neg_instances))
                new_el = el.copy()
                new_el[1] = neg_el
                neg_set = [el_ for el_ in new_el]
                neg_set.append(0)
                lst.append(neg_set)
        train_set = pd.DataFrame(lst, columns = l_features)
        return train_set
    
    def create_test_set(self):
        lst=[]
        mtx = self.df_data_test.values
        possible_instances = list(self.df_data_train.DESTINATION.value_counts().index)
        l_features = list(self.df_data_test.columns)
        l_features.append('Y')
        for el in mtx:
            pos_intances = self.dict_test_data[el[0]]
            tr_pos_intances = self.dict_train_data[el[0]]
            set_neg_instances = set(self.possible_instances) - set(pos_intances) - set(tr_pos_intances)
            pos_set = [el_ for el_ in el]
            pos_set.append(1)
            lst.append(pos_set)
            for i in range(self.Ns):
                neg_el = random.choice(list(set_neg_instances))
                new_el = el.copy()
                new_el[1] = neg_el
                neg_set = [el_ for el_ in new_el]
                neg_set.append(0)
                lst.append(neg_set)
        test_set = pd.DataFrame(lst, columns = l_features)
        return test_set
    
    def features_fm_model(df):
        interactions = df[['CEID', 'DESTINATION']]
        context_features = df[['advanced_purchase', 'nip', 'circle_trip_duration_days', 'round_trip_duration_days',
                              'dpt_day_of_week_1', 'dpt_day_of_week_2', 'dpt_day_of_week_3', 
                              'dpt_day_of_week_4', 'dpt_day_of_week_5', 'dpt_day_of_week_6', 'dpt_day_of_week_7',
                              'departure_month_1', 'departure_month_2', 'departure_month_3',
                              'departure_month_4', 'departure_month_5', 'departure_month_6',
                              'departure_month_7', 'departure_month_8', 'departure_month_9',
                              'departure_month_10', 'departure_month_11', 'departure_month_12',
                              'category_CIRCLE_TRIP', 'category_ONE_WAY', 'category_ROUND_TRIP',
                              'dpt_year_2017', 'dpt_year_2018', 'dpt_year_2019', 'dpt_year_2020']]
        labels = df['Y']
        return interactions.values, context_features.values, labels.values

    def compute_fm_features(tr_interactions, interactions, context_features, labels):
        nb_users = len(np.unique(tr_interactions[:,0]))
        nb_items = len(np.unique(tr_interactions[:,1]))
        X_interaction_features = np.zeros((len(interactions), (nb_users+nb_items)), dtype='uint8')
        for i, el in enumerate(interactions):
            user_index = el[0]
            item_index = n_users + el[1]
            ##
            X_interaction_features[i, user_index] = 1
            X_interaction_features[i, item_index] = 1
        X_fm = np.concatenate([X_interaction_features, context_features], axis=1)
        y_fm = labels
        return X_fm, y_fm
    
class WDL_Data():
    
    
    def __init__(self, ds_path, Ns):  
        ## 
        self.Ns = Ns
        ## 
        self.df_data_test = pd.read_csv(ds_path + 'Travel_data_test.csv')
        self.df_data_train = pd.read_csv(ds_path + 'Travel_data_train.csv')
        self.possible_instances = np.sort(list(df_interaction_train.DEST.value_counts().index))
        ## 
        self.n_talks = len(df_interaction_train.DEST.value_counts())
        self.n_users = len(df_interaction_train.TID.value_counts())
        ## 
        df_train_data = df_data_train.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        df_test_data = df_data_test.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        self.dict_train_data = dict(zip(df_train_data.TID, df_train_data.lists))
        self.dict_test_data = dict(zip(df_test_data.TID, df_test_data.lists))
        
    def create_train_set(self):
        mtx = self.df_data_train.values
        lst=[]
        possible_instances = list(self.df_data_train.DESTINATION.value_counts().index)
        l_features = list(self.df_data_train.columns)
        l_features.append('Y')
        for el in mtx:
            pos_instances = self.dict_train_data[el[0]]
            set_neg_instances = set(self.possible_instances) - set(pos_instances)
            pos_set = [el_ for el_ in el]
            pos_set.append(1)
            lst.append(pos_set)
            for i in range(self.Ns):
                neg_el = random.choice(list(set_neg_instances))
                new_el = el.copy()
                new_el[1] = neg_el
                neg_set = [el_ for el_ in new_el]
                neg_set.append(0)
                lst.append(neg_set)
        train_set = pd.DataFrame(lst, columns = l_features)
        return train_set
    
    def create_test_set(self):
        lst=[]
        mtx = self.df_data_test.values
        possible_instances = list(self.df_data_train.DESTINATION.value_counts().index)
        l_features = list(self.df_data_test.columns)
        l_features.append('Y')
        for el in mtx:
            pos_intances = self.dict_test_data[el[0]]
            tr_pos_intances = self.dict_train_data[el[0]]
            set_neg_instances = set(self.possible_instances) - set(pos_intances) - set(tr_pos_intances)
            pos_set = [el_ for el_ in el]
            pos_set.append(1)
            lst.append(pos_set)
            for i in range(Ns):
                neg_el = random.choice(list(set_neg_instances))
                new_el = el.copy()
                new_el[1] = neg_el
                neg_set = [el_ for el_ in new_el]
                neg_set.append(0)
                lst.append(neg_set)
        test_set = pd.DataFrame(lst, columns = l_features)
        return test_set
    
    def features_wdl_model(df):
        interactions = df[['CEID', 'DESTINATION']]
        features_embed = df[['departure_day', 'booking_class', 'nationality', 'issuanceCountry']]
        features_one_hot = df[['dpt_year_2017', 'dpt_year_2018', 'dpt_year_2019', 'dpt_year_2020',
                                'gender_FEMALE', 'gender_MALE', 'gender_NAN', 'gender_UNDISCLOSED',
                                'dpt_day_of_week_1', 'dpt_day_of_week_2', 'dpt_day_of_week_3', 'dpt_day_of_week_4', 'dpt_day_of_week_5', 'dpt_day_of_week_6', 'dpt_day_of_week_7',
                                'departure_month_1', 'departure_month_2', 'departure_month_3',
                                'departure_month_4', 'departure_month_5', 'departure_month_6',
                                'departure_month_7', 'departure_month_8', 'departure_month_9',
                                'departure_month_10', 'departure_month_11', 'departure_month_12',
                                'category_CIRCLE_TRIP', 'category_ONE_WAY', 'category_ROUND_TRIP',
                                                  ]]
        num_features = df[['age', 'nb_visits', 'nb_diff_des', 'advanced_purchase', 'nip', 'circle_trip_duration_days', 'round_trip_duration_days']]
        labels = df['Y']
        return interactions.values, features_embed.values, features_one_hot.values, num_features.values, labels.values
    
class DKFM_Data():
    
    def __init__(self, ds_path, Ns):  
        ## 
        self.Ns = Ns
        ## 
        self.df_data_test = pd.read_csv(ds_path + 'Travel_data_test.csv')
        self.df_data_train = pd.read_csv(ds_path + 'Travel_data_train.csv')
        self.possible_instances = np.sort(list(df_interaction_train.DEST.value_counts().index))
        ## 
        self.n_talks = len(df_interaction_train.DEST.value_counts())
        self.n_users = len(df_interaction_train.TID.value_counts())
        ## 
        df_train_data = df_data_train.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        df_test_data = df_data_test.groupby('TID')['DEST'].apply(list).reset_index(name='lists')
        self.dict_train_data = dict(zip(df_train_data.TID, df_train_data.lists))
        self.dict_test_data = dict(zip(df_test_data.TID, df_test_data.lists))       
        ## Load embeddings dictionnaries
        with open(ds_path + 'dict_idx_kge_vec.pickle', 'rb') as handle:
            self.dict_iata_kge_vec = pickle.load(handle)
        ##
        with open(ds_path + 'dict_idx_vec.pickle', 'rb') as handle:
            self.dict_iata2vec = pickle.load(handle)
     
    def create_train_set(self):
        lst=[]
        possible_instances = list(self.df_data_train.DESTINATION.value_counts().index)
        l_features = list(self.df_data_train.columns)
        l_features.append('Y')
        for el in self.df_data_train.values:
            pos_intances = self.dict_train_data[el[0]]
            set_neg_instances = set(self.possible_instances) - set(pos_intances)
            pos_set=[]
            pos_set = [el_ for el_ in el]
            pos_set.append(1)
            lst.append(pos_set)
            for i in range(self.Ns):
                neg_el = random.choice(list(set_neg_instances))
                el[1] = neg_el
                neg_set = []
                neg_set = [el_ for el_ in el]
                neg_set.append(0)
                lst.append(neg_set)
        train_set = pd.DataFrame(lst, columns = l_features)
        return train_set
    
    def create_test_set(self):
        lst=[]
        possible_instances = list(self.df_data_train.DESTINATION.value_counts().index)
        l_features = list(self.df_data_test.columns)
        l_features.append('Y')
        for el in df_wdl_test.values:
            pos_intances = self.dict_test_data[el[0]]
            tr_pos_intances = self.dict_train_data[el[0]]
            set_neg_instances = set(self.possible_instances) - set(pos_intances) - set(tr_pos_intances)
            pos_set=[]
            pos_set = [el_ for el_ in el]
            pos_set.append(1)
            lst.append(pos_set)
            for i in range(self.Ns):
                neg_el = random.choice(list(set_neg_instances))
                el[1] = neg_el
                neg_set = []
                neg_set = [el_ for el_ in el]
                neg_set.append(0)
                lst.append(neg_set)
        test_set = pd.DataFrame(lst, columns = l_features)
        return test_set
    
    def features_dkfm_model(df):
        #
        interactions = df[['CEID', 'DESTINATION']]
        #
        features_embed = df[['departure_day', 'booking_class', 'nationality', 'issuanceCountry']]
        #
        features_one_hot = df[[ 'gender_FEMALE', 'gender_MALE', 'gender_NAN', 'gender_UNDISCLOSED']]
        #
        num_features = df[['age', 'nb_visits', 'nb_diff_des']]
        #
        context_features = df[['advanced_purchase', 'nip', 'circle_trip_duration_days', 'round_trip_duration_days',
                              'dpt_year_2017', 'dpt_year_2018', 'dpt_year_2019', 'dpt_year_2020',
                               'dpt_day_of_week_1', 'dpt_day_of_week_2', 'dpt_day_of_week_3', 'dpt_day_of_week_4', 'dpt_day_of_week_5', 'dpt_day_of_week_6', 'dpt_day_of_week_7',
                                'departure_month_1', 'departure_month_2', 'departure_month_3',
                                'departure_month_4', 'departure_month_5', 'departure_month_6',
                                'departure_month_7', 'departure_month_8', 'departure_month_9',
                                'departure_month_10', 'departure_month_11', 'departure_month_12',
                                'category_CIRCLE_TRIP', 'category_ONE_WAY', 'category_ROUND_TRIP']]
        labels = df['Y']
        return interactions.values, context_features.values, features_embed.values, features_one_hot.values, num_features.values, labels.values
    
    def compute_embeddings(tr_interactions):
        tr_kge = np.zeros((len(tr_interactions), 50))
        tr_te = np.zeros((len(tr_interactions), 300))
        for i, el in enumerate(tr_interactions):
            tr_kge[i, :] = self.dict_iata_kge_vec[el[1]]
            tr_te[i, :] = self.dict_iata2vec[el[1]]
        return tr_kge, tr_te