import pandas as pd
import numpy as np
import pickle as pkl
from numpy.linalg import norm
from collections import Counter
from scipy.spatial.distance import euclidean
    
def normalize(x):
    return x/max(norm(x), 10**-12)

class TransE():
    
    def __init__(self, ds_path, e_size, r_size):
        
        ### Load entity Embeddings
        self.ent_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transe/ent_labels.tsv', names=['label'], sep='\t')
        self.dict_ent2idx = dict(zip(list(self.ent_labels.values[:,0]), np.arange(0, len(self.ent_labels), 1)))
        self.ent_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transe/ent_embedding.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')

        ### Load relation Embeddings
        self.rel_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transe/rel_labels.tsv', names=['label'], sep='\t')
        self.dict_rel2idx = dict(zip(list(self.rel_labels.values[:,0]), np.arange(0, len(self.rel_labels), 1)))
        self.rel_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transe/rel_embedding.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')
        self.r_travel = normalize(self.rel_embeddings.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :])
        
    def scoring_function(self, h, t):
        return euclidean(normalize(h) + normalize(self.r_travel), normalize(t))
    
    def eval_function(self, dict_test_positive_user2item, dict_occ_TID, dict_neg_instances, nb_hist, hit):
        hit_10 = 0
        mrr_10 = 0
        K = 0
        for key in dict_test_positive_user2item:
            if key in dict_occ_TID and dict_occ_TID[key] >= nb_hist:
                embd_TID = self.ent_embeddings.values[self.dict_ent2idx[key], :]
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    K = K + 1 
                    l_neg = dict_neg_instances[key]
                    l_score = list()
                    for el in l_pos:
                        airport = self.ent_embeddings.values[self.dict_ent2idx[el]]
                        pos_score = self.scoring_function(embd_TID, airport)
                        ##
                        counter = 0
                        for el_neg in l_neg:
                            airport_neg = self.ent_embeddings.values[self.dict_ent2idx[el_neg]]
                            neg_score = self.scoring_function(embd_TID, airport_neg)
                            if pos_score < neg_score:
                                counter = counter + 1
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter + 1)
                            
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K

        return hit_10, mrr_10, K
    
class TransH():
    
    def __init__(self, ds_path, e_size, r_size):
        
        ### Load entity Embeddings
        self.ent_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transh/ent_labels.tsv', names=['label'], sep='\t')
        self.dict_ent2idx = dict(zip(list(self.ent_labels.values[:,0]), np.arange(0, len(self.ent_labels), 1)))
        self.ent_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transh/ent_embedding.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')

        ### Load relation Embeddings
        self.rel_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transh/rel_labels.tsv', names=['label'], sep='\t')
        self.dict_rel2idx = dict(zip(list(self.rel_labels.values[:,0]), np.arange(0, len(self.rel_labels), 1)))
        self.rel_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transh/rel_embedding.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')
        self.w_projector = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transh/w.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')

        self.r_travel = self.rel_embeddings.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :]
        self.w_travel = normalize(self.w_projector.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :])
        
    def scoring_function(self, h, t):
        #embd_TID = ent_embeddings.values[self.dict_ent2idx[key], :]
        h_wr = h - np.dot(self.w_travel,h)*self.w_travel
        #airport = ent_embeddings.values[self.dict_ent2idx[el]]
        t_wr = t - np.dot(self.w_travel,t)*self.w_travel
        return norm(normalize(h_wr) + normalize(self.r_travel) - normalize(t_wr))
    
    def eval_function(self, dict_test_positive_user2item, dict_occ_TID, dict_neg_instances, nb_hist, hit):
        hit_10 = 0
        mrr_10 = 0
        K = 0
        for key in dict_test_positive_user2item:
            if key in dict_occ_TID and dict_occ_TID[key] >= nb_hist:
                embd_TID = self.ent_embeddings.values[self.dict_ent2idx[key], :]
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    K = K + 1 
                    l_neg = dict_neg_instances[key]
                    l_score = list()
                    for el in l_pos:
                        airport = self.ent_embeddings.values[self.dict_ent2idx[el]]
                        pos_score = self.scoring_function(embd_TID, airport)
                        ##
                        counter = 0
                        for el_neg in l_neg:
                            airport_neg = self.ent_embeddings.values[self.dict_ent2idx[el_neg]]
                            neg_score = self.scoring_function(embd_TID, airport_neg)
                            if pos_score < neg_score:
                                counter = counter + 1
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter + 1)
                            
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K

        return hit_10, mrr_10, K
    
class TransR():
    
    def __init__(self, ds_path, e_size, r_size):
        
        ### Load entity Embeddings
        self.ent_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transr/ent_labels.tsv', names=['label'], sep='\t')
        self.dict_ent2idx = dict(zip(list(self.ent_labels.values[:,0]), np.arange(0, len(self.ent_labels), 1)))
        self.ent_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transr/ent_embedding.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')

        ### Load relation Embeddings
        self.rel_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transr/rel_labels.tsv', names=['label'], sep='\t')
        self.dict_rel2idx = dict(zip(list(self.rel_labels.values[:,0]), np.arange(0, len(self.rel_labels), 1)))
        self.rel_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transr/rel_embedding.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')
        self.rel_matrix = pd.read_csv('../dataset/'+ ds_path + '/embeddings/transr/rel_matrix.tsv', names=['coord_' + str(i) for i in range(e_size*r_size)], sep='\t')

        self.r_travel = self.rel_embeddings.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :]
        self.Mr_travel = self.rel_matrix.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :]
        
    def scoring_function(self, h, t):
        h_Mr = np.matmul(h, self.Mr_travel.reshape(50,50))
        t_Mr = np.matmul(t, self.Mr_travel.reshape(50,50))
        return norm(normalize(h_Mr) + normalize(self.r_travel) - normalize(t_Mr))
    
    def eval_function(self, dict_test_positive_user2item, dict_occ_TID, dict_neg_instances, nb_hist, hit):
        hit_10 = 0
        mrr_10 = 0
        K = 0
        for key in dict_test_positive_user2item:
            if key in dict_occ_TID and dict_occ_TID[key] >= nb_hist:
                embd_TID = self.ent_embeddings.values[self.dict_ent2idx[key], :]
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    K = K + 1 
                    l_neg = dict_neg_instances[key]
                    l_score = list()
                    for el in l_pos:
                        airport = self.ent_embeddings.values[self.dict_ent2idx[el]]
                        pos_score = self.scoring_function(embd_TID, airport)
                        ##
                        counter = 0
                        for el_neg in l_neg:
                            airport_neg = self.ent_embeddings.values[self.dict_ent2idx[el_neg]]
                            neg_score = self.scoring_function(embd_TID, airport_neg)
                            if pos_score < neg_score:
                                counter = counter + 1
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter + 1)
                            
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K

        return hit_10, mrr_10, K
    
class SLM():
    
    def __init__(self, ds_path, e_size, r_size):
        
        self.ent_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/slm/ent_labels.tsv', names=['label'], sep='\t')
        self.dict_ent2idx = dict(zip(list(self.ent_labels.values[:,0]), np.arange(0, len(self.ent_labels), 1)))
        self.ent_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/slm/ent_embedding.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')
        ##
        self.rel_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/slm/rel_labels.tsv', names=['label'], sep='\t')
        self.dict_rel2idx = dict(zip(list(self.rel_labels.values[:,0]), np.arange(0, len(self.rel_labels), 1)))
        self.rel_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/slm/rel_embedding.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')
        ##
        self.mr1 = pd.read_csv('../dataset/'+ ds_path + '/embeddings/slm/mr1.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t').values
        self.mr2 = pd.read_csv('../dataset/'+ ds_path + '/embeddings/slm/mr2.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t').values
        ##
        self.r_travel = self.rel_embeddings.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :]
        
    def scoring_function(self, h, t):
        mr1h = np.matmul(np.reshape(h, (1,50)), self.mr1)
        mr2h = np.matmul(np.reshape(t, (1,50)), self.mr2)
        layer_1 = np.tanh(mr1h+mr2h)
        return -np.sum(normalize(self.r_travel) * layer_1)
    
    def eval_function(self, dict_test_positive_user2item, dict_occ_TID, dict_neg_instances, nb_hist, hit):
        hit_10 = 0
        mrr_10 = 0
        K = 0
        for key in dict_test_positive_user2item:
            if key in dict_occ_TID and dict_occ_TID[key] >= nb_hist:
                embd_TID = self.ent_embeddings.values[self.dict_ent2idx[key], :]
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    K = K + 1 
                    l_neg = dict_neg_instances[key]
                    l_score = list()
                    for el in l_pos:
                        airport = self.ent_embeddings.values[self.dict_ent2idx[el]]
                        pos_score = self.scoring_function(embd_TID, airport)
                        ##
                        counter = 0
                        for el_neg in l_neg:
                            airport_neg = self.ent_embeddings.values[self.dict_ent2idx[el_neg]]
                            neg_score = self.scoring_function(embd_TID, airport_neg)
                            if pos_score < neg_score:
                                counter = counter + 1
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter + 1)
                            
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K

        return hit_10, mrr_10, K
    
class SME_BL():
    
    def __init__(self, ds_path, e_size, r_size):
        
        self.ent_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/ent_labels.tsv', names=['label'], sep='\t')
        self.dict_ent2idx = dict(zip(list(self.ent_labels.values[:,0]), np.arange(0, len(self.ent_labels), 1)))
        self.ent_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/ent_embedding.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')
        ##
        self.rel_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/rel_labels.tsv', names=['label'], sep='\t')
        self.dict_rel2idx = dict(zip(list(self.rel_labels.values[:,0]), np.arange(0, len(self.rel_labels), 1)))
        self.rel_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/rel_embedding.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')
        ##
        self.bu = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/bu.tsv', names=['cord_1'], sep='\t').values
        self.bv = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/bv.tsv', names=['cord_1'], sep='\t').values
        ##
        self.mu1 = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/mu1.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t').values
        self.mu2 = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/mu2.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t').values
        self.mv1 = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/mv1.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t').values
        self.mv2 = pd.read_csv('../dataset/'+ ds_path + '/embeddings/sme_bl/mv2.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t').values
        ##
        self.r_travel = self.rel_embeddings.values[self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>'], :]
        
    def gu_bilinear(self, h):
        mu1h = np.matmul(self.mu1, h.T)  # [k, b]
        mu2r = np.matmul(self.mu2, normalize(self.r_travel.T))  # [k, b]
        return (mu1h * mu2r + self.bu).T  # [b, k]

    def gv_bilinear(self, t):
        mv1t = np.matmul(self.mv1, t.T)  # [k, b]
        mv2r = np.matmul(self.mv2, normalize(self.r_travel.T))  # [k, b]
        return (mv1t * mv2r + self.bv).T  # [b, k]

    def scoring_function(self, h, t):
        norm_h = normalize(h)
        norm_t = normalize(t)
        return np.sum(self.gu_bilinear(norm_h) * self.gv_bilinear(norm_t))
    
    def eval_function(self, dict_test_positive_user2item, dict_occ_TID, dict_neg_instances, nb_hist, hit):
        hit_10 = 0
        mrr_10 = 0
        K = 0
        for key in dict_test_positive_user2item:
            if key in dict_occ_TID and dict_occ_TID[key] >= nb_hist:
                embd_TID = self.ent_embeddings.values[self.dict_ent2idx[key], :]
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    K = K + 1 
                    l_neg = dict_neg_instances[key]
                    l_score = list()
                    for el in l_pos:
                        airport = self.ent_embeddings.values[self.dict_ent2idx[el]]
                        pos_score = self.scoring_function(embd_TID, airport)
                        ##
                        counter = 0
                        for el_neg in l_neg:
                            airport_neg = self.ent_embeddings.values[self.dict_ent2idx[el_neg]]
                            neg_score = self.scoring_function(embd_TID, airport_neg)
                            if pos_score < neg_score:
                                counter = counter + 1
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter + 1)
                            
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K

        return hit_10, mrr_10, K
    
class RotatE():
    
    def __init__(self, ds_path, e_size, r_size, margin):        
        self.ent_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/rotate/ent_labels.tsv', names=['label'], sep='\t')
        self.dict_ent2idx = dict(zip(list(self.ent_labels.values[:,0]), np.arange(0, len(self.ent_labels), 1)))
        self.ent_embeddings_real = pd.read_csv('../dataset/'+ ds_path + '/embeddings/rotate/ent_embeddings_real.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')
        self.ent_embeddings_imag = pd.read_csv('../dataset/'+ ds_path + '/embeddings/rotate/ent_embeddings_imag.tsv', names=['coord_' + str(i) for i in range(e_size)], sep='\t')
        ##
        self.rel_labels = pd.read_csv('../dataset/'+ ds_path + '/embeddings/rotate/rel_labels.tsv', names=['label'], sep='\t')
        self.dict_rel2idx = dict(zip(list(self.rel_labels.values[:,0]), np.arange(0, len(self.rel_labels), 1)))
        self.rel_embeddings = pd.read_csv('../dataset/'+ ds_path + '/embeddings/rotate/rel_embeddings_real.tsv', names=['coord_' + str(i) for i in range(r_size)], sep='\t')
        ##
        self.r_travel = self.dict_rel2idx['<http://data.amadeus.com/ontology/travelTo>']
        ##
        self.margin = margin
        self.embedding_range = (margin + 2.0)/e_size
        
    def embed(self, h, r, t):
        pi = 3.14159265358979323846
        h_e_r = self.ent_embeddings_real.values[h, :]
        h_e_i = self.ent_embeddings_imag.values[h, :]
        r_e_r = self.rel_embeddings.values[r, :]
        t_e_r = self.ent_embeddings_real.values[t, :]
        t_e_i = self.ent_embeddings_imag.values[t, :]
        r_e_r = r_e_r / (self.embedding_range / pi)
        r_e_i = np.sin(r_e_r)
        r_e_r = np.cos(r_e_r)
        return h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i

    def scoring_function(self, h, r, t):
        h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i = self.embed(h, r, t)
        score_r = h_e_r * r_e_r - h_e_i * r_e_i - t_e_r
        score_i = h_e_r * r_e_i + h_e_i * r_e_r - t_e_i
        return -(self.margin - np.sum(score_r**2 + score_i**2, axis=-1))
    
    def eval_function(self, dict_test_positive_user2item, dict_occ_TID, dict_neg_instances, nb_hist, hit):
        pos_score_l = list()
        neg_score_l = list()
        hit_10 = 0
        mrr_10 = 0
        K = 0
        for key in dict_test_positive_user2item:
            if key in dict_occ_TID and dict_occ_TID[key] >=nb_hist:
                K = K + 1 
                embd_TID_ID = self.dict_ent2idx[key]
                l_pos = dict_test_positive_user2item[key]
                if key in dict_neg_instances:
                    l_neg = dict_neg_instances[key]
                    l_score = list()
                    for el in l_pos:
                        airport = self.dict_ent2idx[el]
                        pos_score = self.scoring_function(embd_TID_ID, self.r_travel, airport)
                        ##
                        counter = 0
                        for el_neg in l_neg:
                            airport_neg = self.dict_ent2idx[el_neg]
                            neg_score = self.scoring_function(embd_TID_ID, self.r_travel, airport_neg)
                            if pos_score <= neg_score:
                                counter = counter + 1
                        if counter>=len(l_neg)-hit:
                            hit_10 = hit_10 + 1
                            mrr_10 = mrr_10 + 1/(len(l_neg) - counter+1)
        hit_10 = hit_10/K
        mrr_10 = mrr_10/K

        return hit_10, mrr_10, K