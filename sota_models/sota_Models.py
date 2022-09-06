import numpy as np
import math
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Multiply, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import pickle
import random
import tqdm

### TF Implementation of Matrix Factorization algotihn with Adam optimizer 
def get_model_MF(num_users, num_items, factors=64, lr=0.0001):
    d = factors
    
    user_input_MF = Input(shape=(1,), dtype='int32', name='user_input')
    item_input_MF = Input(shape=(1,), dtype='int32', name='item_input')

    user_embedding_MF = Embedding(input_dim=num_users, output_dim=d, name='user_embedding')(user_input_MF)
    item_embedding_MF = Embedding(input_dim=num_items, output_dim=d, name='item_embedding')(item_input_MF)

    user_latent_MF = Flatten()(user_embedding_MF)
    item_latent_MF = Flatten()(item_embedding_MF)
    
    dot = Dot(axes=1)([user_latent_MF, item_latent_MF])
    
    prediction_MF = Dense(units=1, activation='sigmoid', name='prediction')(dot)

    MF = Model(inputs=[user_input_MF, item_input_MF], outputs=prediction_MF)
    MF.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    return MF

### TF Implementation of Generalized Matrix Factorization model
def get_model_GMF(num_users, num_items, factors=128, lr=0.0001):
    d=factors
    user_input_GMF = Input(shape=(1,), dtype='int32', name='user_input_GMF')
    item_input_GMF = Input(shape=(1,), dtype='int32', name='item_input_GMF')

    user_embedding_GMF = Embedding(input_dim=num_users, output_dim=d, name='user_embedding_GMF')
    item_embedding_GMF = Embedding(input_dim=num_items, output_dim=d, name='item_embedding_GMF')

    user_latent_GMF = Flatten()(user_embedding_GMF(user_input_GMF))
    item_latent_GMF = Flatten()(item_embedding_GMF(item_input_GMF))

    mul = Multiply()([user_latent_GMF, item_latent_GMF]) # len = factors

    prediction_GMF = Dense(units=1, activation='sigmoid', name='prediction')(mul)

    GMF = Model(inputs=[user_input_GMF, item_input_GMF], outputs=prediction_GMF)
    GMF.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    return GMF

### TF Implementation of MLP model 
def get_model_MLP(num_users, num_items, num_layers=6, factors=8, lr=0.0001):
    if num_layers==0:
        d = int(factors/2)
    else:
        d = int((2**(num_layers-2))*factors)
        
    user_input_MLP = Input(shape=(1,), dtype='int32', name='user_input_MLP')
    item_input_MLP = Input(shape=(1,), dtype='int32', name='item_input_MLP')
    
    user_embedding_MLP = Embedding(input_dim=num_users, output_dim=d, name='user_embedding_MLP')(user_input_MLP)
    item_embedding_MLP = Embedding(input_dim=num_items, output_dim=d, name='item_embedding_MLP')(item_input_MLP)
    
    user_latent_MLP = Flatten()(user_embedding_MLP)
    item_latent_MLP = Flatten()(item_embedding_MLP)
    
    concatenation = Concatenate()([user_latent_MLP, item_latent_MLP])
    output = concatenation
    layer_name = 0
    for i in range(num_layers-1,-1,-1):
        layer = Dense(units=(2**i)*factors, activation='tanh', name='layer%d' %(layer_name+1))
        output = layer(output)
        layer_name += 1
    prediction_MLP = Dense(units=1, activation='sigmoid', name='prediction_MLP')(output)
    MLP = Model(inputs=[user_input_MLP, item_input_MLP], outputs=prediction_MLP)
    MLP.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    return MLP

### Neural collaborative filtering model
def get_model_NCF(num_users, num_items, num_layers_MLP_part=4, factors=32, lr=0.0001):
    assert (factors%2)==0
    if num_layers_MLP_part==0:
        d_MLP = int(factors/4)
    else:
        d_MLP = (2**(num_layers_MLP_part-3))*factors

    user_input = Input(shape=(1,), dtype='int32', name='user_input_NeuMF')
    item_input = Input(shape=(1,), dtype='int32', name='item_input_NeuMF')

    ## MLP part
    user_embedding_MLP = Embedding(input_dim=num_users, output_dim=d_MLP, name='user_embedding_MLP')(user_input)
    item_embedding_MLP = Embedding(input_dim=num_items, output_dim=d_MLP, name='item_embedding_MLP')(item_input)

    user_latent_MLP = Flatten()(user_embedding_MLP)
    item_latent_MLP = Flatten()(item_embedding_MLP)

    concatenation_embeddings = Concatenate()([user_latent_MLP, item_latent_MLP])
    
    output_MLP = concatenation_embeddings  
    layer_name = 0
    for i in range(num_layers_MLP_part-2,-2,-1):
        layer = Dense(units=(2**i)*factors, activation='tanh', name='layer%d' %(layer_name+1))
        output_MLP = layer(output_MLP)
        layer_name += 1
    
    d_GMF = int(factors/2)
    ## GMF part
    user_embedding_GMF = Embedding(input_dim=num_users, output_dim=d_GMF, name='user_embedding_GMF')
    item_embedding_GMF = Embedding(input_dim=num_items, output_dim=d_GMF, name='item_embedding_GMF')

    user_latent_GMF = Flatten()(user_embedding_GMF(user_input))
    item_latent_GMF = Flatten()(item_embedding_GMF(item_input))

    mul = Multiply()([user_latent_GMF, item_latent_GMF])

    concatenation_of_models = Concatenate(name='final_concatenation')([mul, output_MLP]) # len = factors
    prediction_NeuMF = Dense(units=1, activation='sigmoid', name='prediction')(concatenation_of_models)

    NeuMF = Model(inputs=[user_input, item_input], outputs=prediction_NeuMF)
    NeuMF.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    return NeuMF

## Facotrization machines model
class FMLayer(tf.keras.layers.Layer):
    def __init__(self, nb_num_features, hidden_dim):
        super(MyLayer, self).__init__()

        #your variable goes here
        # bias and weights
        self.w0 = tf.Variable(tf.random.normal([1], stddev=0.01), trainable=True)
        self.W = tf.Variable(tf.random.normal([nb_num_features], stddev=0.01), trainable=True)

        # interaction factors, randomly initialized 
        self.V = tf.Variable(tf.random.normal([hidden_dim, nb_num_features], stddev=0.01), trainable=True)
            
    def call(self, X):

        # your mul operation goes here
        linear_terms = tf.add(self.w0, tf.reduce_sum(tf.multiply(self.W, X), 1, keepdims=True))
        pair_interactions = (tf.multiply(0.5,
                            tf.reduce_sum(
                                tf.subtract(
                                    tf.pow( tf.matmul(X, tf.transpose(self.V)), 2),
                                    tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(self.V , 2)))),
                                1, keepdims=True)))
        
        y_hat = tf.math.sigmoid(tf.add(linear_terms, pair_interactions))
                        
        return y_hat

def get_model_FM(nb_num_features, hidden_dim=16, lr = 0.00005):
    
    #tr_interactions, tr_features_embed, tr_features_one_hot, tr_num_features, tr_labels
    
    ### INTERACTIONS INPUTS
    X = Input(shape=(nb_num_features,), dtype='float32', name='input_features')

    #model = tf.keras.models.Sequential()
    FM = FMLayer(nb_num_features, hidden_dim)
    
    y_hat = FM(X)
    
    FM = Model(inputs=X, outputs=y_hat)
    FM.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    
    return FM

### TF implementation of WDL model

def get_model_WDL(num_users, num_items, nb_num_features, nb_features_one_hot, nb_dpt_days, nb_nat, nb_iss, nb_bkg,
                  hidden_dim_user=128, hidden_dim_item=32, dim_layer=128, nb_layers=4, lr = 0.00005):
        
    ### INTERACTIONS INPUTS
    user_input = Input(shape=(1,), dtype='int32', name='user_input_WDL')
    item_input = Input(shape=(1,), dtype='int32', name='item_input_WDL')
    
    ### EMBEDDING FEATURES INPUT
    dpt_day_input = Input(shape=(1,), dtype='int32', name='dpt_day_input_WDL')
    nat_input = Input(shape=(1,), dtype='int32', name='nat_input_WDL')
    iss_input = Input(shape=(1,), dtype='int32', name='iss_country_input_WDL')
    bkg_input = Input(shape=(1,), dtype='int32', name='kkg_class_input_WDL')
    
    ### ONE HOT ENCODED FEATURES
    features_one_hot = Input(shape=(nb_features_one_hot,), dtype='float32', name='features_one_hot_WDL')
    
    ### NUMERICAL FEATURES
    num_features = Input(shape=(nb_num_features,), dtype='float32', name='features_num_WDL')
    
    ## Embeddings part
    user_embedding_W = Embedding(input_dim=num_users, output_dim=hidden_dim_user, name='user_embedding_WDL')(user_input)
    item_embedding_W = Embedding(input_dim=num_items, output_dim=hidden_dim_item, name='item_embedding_WDL')(item_input)
    dpt_day_input_W = Embedding(input_dim=nb_dpt_days+1, output_dim=8, name='dpt_day_embedding')(dpt_day_input)
    nat_input_W = Embedding(input_dim=nb_nat, output_dim=16, name='nat_embedding_WDL')(nat_input)
    iss_input_W = Embedding(input_dim=nb_iss, output_dim=16, name='iss_embedding')(iss_input)
    bkg_input_W = Embedding(input_dim=nb_bkg, output_dim=8, name='bkg_embedding_WDL')(bkg_input)
    
    ## Flatten embeddings
    user_latent = Flatten()(user_embedding_W)
    item_latent = Flatten()(item_embedding_W)
    dpt_day_latent = Flatten()(dpt_day_input_W)
    nat_latent = Flatten()(nat_input_W)
    iss_latent = Flatten()(iss_input_W)
    bkg_latent = Flatten()(bkg_input_W)
    
    print(num_features.shape)
    
    concat = Concatenate()([user_latent, item_latent, dpt_day_latent, nat_latent, iss_latent, bkg_latent])
    
    print(concat.shape)
    
    output_hidden = Dense(units=dim_layer, activation='tanh', name='layer_0')(concat)
        
    print(output_hidden.shape)
    
    layer_name = 0
    for i in range(nb_layers):
        layer = Dense(units=dim_layer/(2*(i+1)), activation='tanh', name='layer_%d' %(layer_name+1))
        output_hidden = layer(output_hidden)
        layer_name += 1
        
    wd = Concatenate()([output_hidden, num_features, features_one_hot])
    
    #layer_name = 0
    for i in range(nb_layers):
        layer = Dense(units=dim_layer/(2*(i+1)), activation='tanh', name='layer_%d' %(layer_name+1))
        wd = layer(wd)
        layer_name += 1
        
    prediction_WDL = Dense(units=1, activation='sigmoid', name='prediction')(wd)
    
    print(prediction_WDL.shape)
    
    WDL = Model(inputs=[user_input, item_input, dpt_day_input, nat_input, iss_input, bkg_input, features_one_hot, num_features], outputs=prediction_WDL)
    WDL.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    
    return WDL


### TF implementation of  WDL model
class NFM_Pool_Layer(tf.keras.layers.Layer):
    def __init__(self, nb_context_features, factors):
        super(NFM_Pool_Layer, self).__init__()
        # interaction factors, randomly initialized 
        initializer = tf.keras.initializers.GlorotNormal()
        self.V = tf.Variable(initializer(shape=(factors, nb_context_features)), trainable=True)
            
    def call(self, X_ctxt):
        # Calculate output with FM equation
        pool_layer = tf.multiply(0.5,
                            tf.subtract(
                                tf.pow( tf.matmul(X_ctxt, tf.transpose(self.V)), 2),
                                tf.matmul(tf.pow(X_ctxt, 2), tf.transpose(tf.pow(self.V, 2)))))
        
        return pool_layer

def get_model_DKFM(num_users, num_items, nb_num_features, nb_features_one_hot, nb_context_features, 
                   kge_dim, te_dim,
                   nb_dpt_days, nb_nat, nb_iss, nb_bkg, factors=16,
                  hidden_dim=128, dim_layer=256, nb_layers=3, lr = 0.00005):
        
    ### INTERACTIONS INPUTS
    user_input = Input(shape=(1,), dtype='int32', name='user_input_DKFM')
    item_input = Input(shape=(1,), dtype='int32', name='item_input_DKFM')
    
    ### EMBEDDING FEATURES INPUT
    dpt_day_input = Input(shape=(1,), dtype='int32', name='dpt_day_input_DKFM')
    nat_input = Input(shape=(1,), dtype='int32', name='nat_input_DKFM')
    iss_input = Input(shape=(1,), dtype='int32', name='iss_country_input_DKFM')
    bkg_input = Input(shape=(1,), dtype='int32', name='kkg_class_input_DKFM')
    
    ### ONE HOT ENCODED FEATURES
    features_one_hot = Input(shape=(nb_features_one_hot,), dtype='float32', name='features_one_hot_DKFM')
    
    ### NUMERICAL FEATURES
    num_features = Input(shape=(nb_num_features,), dtype='float32', name='features_num_DKFM')
    
    ### Contextual Features
    context_features = Input(shape=(nb_context_features,), dtype='float32', name='context_features_DKFM')
    
    ### KGE Embeddings
    kge = Input(shape=(kge_dim,), dtype='float32', name='KGE_DKFM')
    
    ### Textual Embeddings
    te = Input(shape=(te_dim,), dtype='float32', name='TE_DKFM')
    
    ## Embeddings part
    user_embedding_W = Embedding(input_dim=num_users, output_dim=hidden_dim, name='user_embedding_DKFM')(user_input)
    item_embedding_W = Embedding(input_dim=num_items, output_dim=hidden_dim, name='item_embedding_DKFM')(item_input)
    dpt_day_input_W = Embedding(input_dim=nb_dpt_days+1, output_dim=16, name='dpt_day_embedding_DKFM')(dpt_day_input)
    nat_input_W = Embedding(input_dim=nb_nat, output_dim=16, name='nat_embedding_DKFM')(nat_input)
    iss_input_W = Embedding(input_dim=nb_iss, output_dim=16, name='iss_embedding_DKFM')(iss_input)
    bkg_input_W = Embedding(input_dim=nb_bkg, output_dim=16, name='bkg_embedding_DKFM')(bkg_input)
    
    ## Flatten embeddings
    user_latent = Flatten()(user_embedding_W)
    item_latent = Flatten()(item_embedding_W)
    dpt_day_latent = Flatten()(dpt_day_input_W)
    nat_latent = Flatten()(nat_input_W)
    iss_latent = Flatten()(iss_input_W)
    bkg_latent = Flatten()(bkg_input_W)
    
    ## FM part 
    FM = NFM_Pool_Layer(nb_context_features, factors)
    pool_layer = FM(context_features)
    
    print(num_features.shape)
    
    concat = Concatenate()([user_latent, item_latent, dpt_day_latent, nat_latent, iss_latent, bkg_latent, kge, te, pool_layer])
    
    print(concat.shape)
    
    output_hidden = Dense(units=dim_layer, activation='tanh', name='layer_0')(concat)
        
    print(output_hidden.shape)
    
    layer_name = 0
    for i in range(nb_layers):
        layer = Dense(units=dim_layer/(2*(i+1)), activation='tanh', name='layer_%d' %(layer_name+1))
        output_hidden = layer(output_hidden)
        layer_name += 1
        
    wd = Concatenate()([output_hidden, num_features, features_one_hot]) 
        
    for i in range(nb_layers):
        layer = Dense(units=dim_layer/(2*(i+1)), activation='tanh', name='layer_%d' %(layer_name+1))
        wd = layer(wd)
        layer_name += 1    
    
        
    prediction_DKFM = Dense(units=1, activation='sigmoid', name='prediction')(wd)
    
    print(prediction_DKFM.shape)
    
    DKFM = Model(inputs=[user_input, item_input, dpt_day_input, nat_input, iss_input, bkg_input, features_one_hot, num_features, context_features, kge, te], outputs=prediction_DKFM)
    DKFM.compile(optimizer=tf.optimizers.Adam(lr=lr), loss=binary_crossentropy)
    
    return DKFM

