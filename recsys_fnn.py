########################################
## import packages
########################################

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import numpy as np

import sys

import pandas as pd


from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, merge
from keras.layers.merge import concatenate, dot, add
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import AlphaDropout
from keras.callbacks import EarlyStopping, CSVLogger, TerminateOnNaN
from keras.regularizers import l2
from keras.initializers import RandomNormal, Constant, lecun_normal
from keras.optimizers import RMSprop, Adam, SGD
from keras.metrics import mse
from keras import regularizers

from keras import backend as K

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

_SESSION = None
_IS_CALLED = 0


def set_session(session):
    """Sets the global TensorFlow session.
    # Arguments
        session: A TF Session.
    """
    global _SESSION
    _SESSION = session


########################################
## Hyperparameters
########################################

emb_bias_regularization = 1e-4
dense_regularization = 1e-4
factors = 100
batchsize=256
dense_units = 128

network=sys.argv[1]
learning_rate = float(sys.argv[2])
optimizer = sys.argv[3]
act_function = sys.argv[4]
dropout = float(sys.argv[5])
num_epochs = int(sys.argv[6])
xi = float(sys.argv[7])
dataset = str(sys.argv[8])
split = int(sys.argv[9])


if optimizer=="SGD":
    optimizer = optimizer+'('+str(learning_rate)+', momentum='+str(xi)+')'
else:
    optimizer = optimizer+'('+str(learning_rate)+')'

########################################
## Print Hyperparameters
########################################

print "optimizer= ", str(optimizer)
print "Learning_Rate= ", learning_rate
print "Dropout= ", dropout
print "Activation_function= ", act_function
print "Emb/Bias_Regularization= ", emb_bias_regularization
print "preds_Regularization= ", dense_regularization
print "factors= ", factors
print "Batch_Size= ", batchsize
print "Number_of_epochs= ", num_epochs

########################################
## define metrics
########################################

def mae(y_true, y_pred):
    s_s = K.mean(K.abs(y_pred - y_true), axis=-1)
    return s_s

graph = tf.Graph()

with graph.as_default():

    with tf.Session(graph=graph) as sess:

        ########################################
        ## load the data
        ########################################

        train = pd.read_csv('./my_data/'+dataset+'/u'+str(split)+'.train', sep='\t')
        #train = pd.read_csv('train.csv', sep='\t')
        uid = train.msno
        sid = train.song_id
        target = train.target

        test = pd.read_csv('./my_data/'+dataset+'/u'+str(split)+'.test', sep='\t')
        #test = pd.read_csv('test.csv', sep='\t')
        uid_test = test.msno
        sid_test = test.song_id
        target_test = test.target

        ########################################
        ## encoding
        ########################################

        u_cnt = int(max(uid.max(), uid_test.max()) + 1)
        s_cnt = int(max(sid.max(), sid_test.max()) + 1)

        print(u_cnt,s_cnt)

        ########################################
        ## train-validation split
        ########################################

        trn_cnt = int(len(train))
        uid_trn = uid
        uid_val = uid_test
        sid_trn = sid
        sid_val = sid_test
        target_trn = target
        target_val = target_test

        ########################################
        ## useful functions
        ########################################

        def create_global_mean(inp, n_in):

            x = Embedding(n_in, 1, embeddings_initializer=Constant(value=target.mean()), embeddings_regularizer=None, input_length=1, trainable=False)(inp)

            return Reshape((1,))(x)

        def create_bias(inp, n_in, reg, layer_name):

            x = Embedding(n_in, 1, embeddings_initializer="zeros",
        	embeddings_regularizer=l2(reg),
		input_length=1, trainable=True,name=layer_name)(inp)
            return Reshape((1,), name=layer_name+'_reshaped')(x)

        ########################################
        ## define the model
        ########################################

        def single_nn(opt, dropout, a_func, reg_all, preds_reg, embed_dim, dense_units):

            user_embeddings = Embedding(u_cnt,
                    embed_dim,
                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1),
                    embeddings_regularizer=l2(reg_all),
                    input_length=1,
                    trainable=True, name='user_embeddings')

            song_embeddings = Embedding(s_cnt,
                    embed_dim,
                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1),
                    embeddings_regularizer=l2(reg_all),
                    input_length=1,
                    trainable=True,name='item_embeddings')


            uid_input = Input(shape=(1,), dtype='int32')
            embedded_usr = user_embeddings(uid_input)
            embedded_usr = Reshape((embed_dim,),name='user_embeddings_reshaped')(embedded_usr)
            ub = create_bias(uid_input, u_cnt, reg_all,'user_biases')

            sid_input = Input(shape=(1,), dtype='int32')
            embedded_song = song_embeddings(sid_input)
            embedded_song = Reshape((embed_dim,),name='item_embeddings_reshaped')(embedded_song)
            mb = create_bias(sid_input, s_cnt, reg_all,'item_biases')

            preds = dot([embedded_usr, embedded_song], axes=1)

            preds = concatenate([embedded_usr, embedded_song, preds], name='concatenated_embeddings_all')

            if (a_func == 'relu') or (a_func == 'elu'):
                preds = Dense(1, use_bias=True, activation=a_func, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), kernel_regularizer=regularizers.l2(preds_reg), bias_initializer=RandomNormal(mean=0.0, stddev=0.1),bias_regularizer=regularizers.l2(preds_reg), name='main_output')(preds)
                preds=Dropout(dropout)(preds)
            elif a_func == 'selu':
                preds = Dense(1, activation=a_func, kernel_initializer=lecun_normal(), name='main_output')(preds)
                preds=AlphaDropout(dropout)(preds)

            preds = add([ub, preds])
            preds = add([mb, preds])

            model = Model(inputs=[uid_input, sid_input], outputs=preds)
            opt = eval(opt)
            model.compile(loss='mse', optimizer=opt, metrics=[mae])

            return model

        def single_nn_gm(opt, dropout, a_func, reg_all, preds_reg, embed_dim, dense_units):

            user_embeddings = Embedding(u_cnt,
                    embed_dim,
                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1),
                    embeddings_regularizer=l2(reg_all),
                    input_length=1,
                    trainable=True, name='user_embeddings')

            song_embeddings = Embedding(s_cnt,
                    embed_dim,
                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1),
                    embeddings_regularizer=l2(reg_all),
                    input_length=1,
                    trainable=True,name='item_embeddings')


            uid_input = Input(shape=(1,), dtype='int32')
            embedded_usr = user_embeddings(uid_input)
            embedded_usr = Reshape((embed_dim,),name='user_embeddings_reshaped')(embedded_usr)
            ub = create_bias(uid_input, u_cnt, reg_all,'user_biases')

            sid_input = Input(shape=(1,), dtype='int32')
            embedded_song = song_embeddings(sid_input)
            embedded_song = Reshape((embed_dim,),name='item_embeddings_reshaped')(embedded_song)
            mb = create_bias(sid_input, s_cnt, reg_all,'item_biases')

            global_mean=create_global_mean(uid_input,u_cnt)

            preds = dot([embedded_usr, embedded_song], axes=1)

            preds = add([global_mean, preds])

            preds = concatenate([embedded_usr, embedded_song, preds], name='concatenated_embeddings_all')

            if (a_func == 'relu') or (a_func == 'elu'):
                preds = Dense(1, use_bias=True, activation=a_func, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), kernel_regularizer=regularizers.l2(preds_reg), bias_initializer=RandomNormal(mean=0.0, stddev=0.1),bias_regularizer=regularizers.l2(preds_reg))(preds)
                preds=Dropout(dropout)(preds)
            elif a_func == 'selu':
                preds = Dense(1, activation=a_func, kernel_initializer=lecun_normal(), name='main_output')(preds)
                preds=AlphaDropout(dropout)(preds)

            preds = add([ub, preds])
            preds = add([mb, preds])

            model = Model(inputs=[uid_input, sid_input], outputs=preds)
            opt = eval(opt)
            model.compile(loss='mse', optimizer=opt, metrics=[mae])

            return model

        def double_nn(opt, dropout, a_func, reg_all, preds_reg, embed_dim, dense_units):

            user_embeddings = Embedding(u_cnt,
                    embed_dim,
                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1),
                    embeddings_regularizer=l2(reg_all),
                    input_length=1,
                    trainable=True, name='user_embeddings')

            song_embeddings = Embedding(s_cnt,
                    embed_dim,
                    embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1),
                    embeddings_regularizer=l2(reg_all),
                    input_length=1,
                    trainable=True,name='item_embeddings')


            uid_input = Input(shape=(1,), dtype='int32')
            embedded_usr = user_embeddings(uid_input)
            embedded_usr = Reshape((embed_dim,),name='user_embeddings_reshaped')(embedded_usr)
            ub = create_bias(uid_input, u_cnt, reg_all,'user_biases')


            sid_input = Input(shape=(1,), dtype='int32')
            embedded_song = song_embeddings(sid_input)
            embedded_song = Reshape((embed_dim,),name='item_embeddings_reshaped')(embedded_song)
            mb = create_bias(sid_input, s_cnt, reg_all,'item_biases')

            preds = concatenate([embedded_usr, embedded_song],name='concatenated_embeddings_all')

            if (a_func == 'relu') or (a_func == 'elu'):
                preds = Dense(dense_units, use_bias=True, activation=a_func, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), kernel_regularizer=regularizers.l2(preds_reg), bias_initializer=RandomNormal(mean=0.0, stddev=0.1),bias_regularizer=regularizers.l2(preds_reg))(preds)
                preds=Dropout(dropout)(preds)
                preds = Dense(1, use_bias=True, activation=a_func, kernel_initializer=RandomNormal(mean=0.0, stddev=0.1), kernel_regularizer=regularizers.l2(preds_reg), bias_initializer=RandomNormal(mean=0.0, stddev=0.1),bias_regularizer=regularizers.l2(preds_reg))(preds)
                preds=Dropout(dropout)(preds)
            elif a_func == 'selu':
                preds = Dense(dense_units, activation=a_func, kernel_initializer=lecun_normal(), name='main_hidden')(preds)
                preds=AlphaDropout(dropout)(preds)
                preds = Dense(1, activation=a_func, kernel_initializer=lecun_normal(), name='main_output')(preds)
                preds=AlphaDropout(dropout)(preds)

            preds = add([ub, preds])
            preds = add([mb, preds])

            model = Model(inputs=[uid_input, sid_input], outputs=preds)
            opt = eval(opt)
            model.compile(loss='mse', optimizer=opt, metrics=[mae])

            return model

        ########################################
        ## train the model
        ########################################

        model = eval(network+'(optimizer, dropout, act_function, emb_bias_regularization, dense_regularization, factors, dense_units)')

        model.summary()

        if network=="single_nn":
            filename = 'results/results_'+network+'_opt_'+str(optimizer)+'_dr_'+str(dropout)+'_r_all_'+str(emb_bias_regularization)+'_r_pr_'+str(dense_regularization)+'_'+str(act_function)+'_'+str(dense_units)+'_'+str(dataset)+'_'+str(split)+'.csv'
            net_type = "Single Layer Feedforward Neural Network"
        elif network=="single_nn_gm":
            filename = 'results/results_'+network+'_opt_'+str(optimizer)+'_dr_'+str(dropout)+'_r_all_'+str(emb_bias_regularization)+'_r_pr_'+str(dense_regularization)+'_'+str(act_function)+'_'+str(dense_units)+'_'+str(dataset)+'_'+str(split)+'.csv'
            net_type = "Single Layer Feedforward Neural Network with Global Mean"
        elif network=="double_nn":
            filename = 'results/results_'+network+'_opt_'+str(optimizer)+'_dr_'+str(dropout)+'_r_all_'+str(emb_bias_regularization)+'_r_pr_'+str(dense_regularization)+'_'+str(act_function)+'_'+str(dense_units)+'_'+str(dataset)+'_'+str(split)+'.csv'
            net_type = "Double Layer Feedforward Neural Network"


        ########################################
        ## callbacks
        ########################################

        csv_logger = CSVLogger(filename)
        call_nan = TerminateOnNaN()
        early_stopping = EarlyStopping(monitor='val_mae', patience=10)

        model.fit([uid_trn, sid_trn], target_trn, validation_data=([uid_val, sid_val], target_val), epochs=num_epochs, batch_size=batchsize, shuffle=False, callbacks=[csv_logger, call_nan, early_stopping])

        df = pd.read_csv(filename)

        minimum_mae = df['val_mae'].min(axis=0)
        minimum_index = df['val_mae'].idxmin(axis=0) + 1

        print "Best result: ",  minimum_mae
        print "-------------------------------------------------------------"
        print net_type+",", minimum_mae, ",",sys.argv[3], ",",learning_rate, ",",xi, ",",dropout, ",",emb_bias_regularization, ",",dense_regularization, ",",act_function, ",",minimum_index, ",", dataset, ",", split
