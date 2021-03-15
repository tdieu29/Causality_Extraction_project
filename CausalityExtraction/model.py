import tensorflow as tf 
import keras
from pathlib import Path
import numpy as np
import os
import random as rn
from keras import backend as K 
import h5py
from keras.layers import *
import math
from keras.models import Model
from keras import optimizers
from keras.callbacks import *

from MHSA import MultiHeadSelfAttention
from ChainCRF import ChainCRF
from tag2triplet import *

from CausalityExtraction import configurations

import wandb
from wandb.keras import WandbCallback


# Constants and arguments
index_path = configurations.INDEX_DIR
embedding_path = configurations.EMBEDDING_DIR
train_path = configurations.TRAIN_DIR
test_path = configurations.TEST_DIR
save_path = configurations.LOGS_DIR

MAX_WLEN = 58
MAX_CLEN = 23
CHAR_SIZE = 70
EXTVEC_DIM = 300
FLAIR_DIM = 4096
CHAR_DIM = 30
NUM_CHAR_CNN_FILTER = 30
CHAR_CNN_KERNEL_SIZE = 3
NUM_CLASS = 7


class MaskConv1D(Conv1D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
  
    def compute_mask(self, inputs, mask=None):
        return mask
  
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super().call(inputs)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, x, x_flair, x_char, y, batch_size, pred=False):
        self.list_IDs = list_IDs
        self.x = x
        self.x_flair = x_flair
        self.x_char = x_char
        self.y = y
        self.batch_size = batch_size
        self.pred = pred
  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))
  
    def __getitem__(self, index):
        'Generate one batch of data'
        list_IDs_temp = self.list_IDs[index * self.batch_size:(index+1) * self.batch_size]
        return self.__data_generation(list_IDs_temp)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        x = []

        if self.pred:
            maxlen = MAX_WLEN
            maxlen_c = MAX_CLEN
        else:
            maxlen = max([len(np.where(self.x[ID] != 0)[0]) for ID in list_IDs_temp])
            maxlen_c = max([len(np.where(self.x_char[ID][_] != 0)[0]) 
                                for _ in range(maxlen) for ID in list_IDs_temp])
      
        x.append(np.zeros((self.batch_size, maxlen)))
        for i, ID in enumerate(list_IDs_temp):
            x[0][i] = self.x[ID][:maxlen]
    

        x_flair = np.zeros((self.batch_size, maxlen, FLAIR_DIM))
        for i, ID in enumerate(list_IDs_temp):
            x_flair[i] = self.x_flair[ID][:maxlen]
        x.append(x_flair)
    

        x_char = np.zeros((self.batch_size, maxlen, maxlen_c))
        for i, ID in enumerate(list_IDs_temp):
            x_char[i] = self.x_char[ID][:maxlen][:, :maxlen_c]
        x.append(x_char)

        if self.pred:
            return x
    
        y = np.zeros((self.batch_size, maxlen, 1))
        for i, ID in enumerate(list_IDs_temp):
            y[i] = self.y[ID][:maxlen]
    
        return x, y


class Data:
    def __init__(self, config):
        fp1 = str(Path(index_path, 'index_w.pkl'))
        self.word2index, self.index2word = pickle.load(open(fp1, 'rb'))
        
        fp2 = str(Path(embedding_path, 'extvec_embedding.npy'))
        self.embedding = np.load(open(fp2, 'rb'))
    
        self.VOCAB_SIZE = len(self.word2index)

        with h5py.File(str(Path(train_path, 'train.h5')), 'r') as fh:
            self.xTrain = fh['xTrain'][:]
            self.yTrain = fh['yTrain'][:]
    
        with h5py.File(str(Path(test_path, 'test.h5')), 'r') as fh:
            self.xTest = fh['xTest'][:]
            self.yTest = fh['yTest'][:]
    
        self.config = config

    def cross_validation(self):
        """
        Return the data from cross validation
        """
        # Flair embedding
        h5f = h5py.File(str(Path(embedding_path, 'flair.h5')), 'r')
        train_flair = h5f['xTrain_flair']

        # Char embedding   
        h5f_te = h5py.File(str(Path(train_path, 'train.h5')), 'r')
        train_char = h5f_te['xTrain_c']

        # valIdx and trainIdx
        l = [i for i in range(len(self.xTrain))]
        trainIdx = shuffle(l, random_state = self.config.seed)
      
        valIdx = trainIdx[self.config.k_fold*(len(trainIdx)//10):(self.config.k_fold+1)*(len(trainIdx)//10)]
        trainIdx = [i for i in trainIdx if i not in valIdx] 

        # eval_x, eval_y
        eval_x, eval_y = self.xTrain[valIdx], self.yTrain[valIdx]

        # Data generators
        training_generator = DataGenerator(trainIdx, 
                                            x=self.xTrain,
                                            x_flair=train_flair,
                                            x_char=train_char,
                                            y=self.yTrain,
                                            batch_size=self.config.batch_size)
      
        validation_generator = DataGenerator(valIdx,
                                            x=self.xTrain,
                                            x_flair=train_flair,
                                            x_char=train_char,
                                            y=self.yTrain,                                                                      
                                            batch_size=self.config.batch_size)                                                         
                                       
        predict_generator = DataGenerator(valIdx,
                                        x=self.xTrain,
                                        x_flair=train_flair,
                                        x_char=train_char,
                                        y=self.yTrain,
                                        batch_size=self.config.batch_size,     
                                        pred=True)

        return eval_x, eval_y, training_generator, validation_generator, predict_generator

    def train(self):
        """
        Return the data of train
        """
        # Flair embedding
        h5f = h5py.File(str(Path(embedding_path, 'flair.h5')), 'r')
        train_flair = h5f['xTrain_flair']
        test_flair = h5f['xTest_flair']

        # Char embedding
        h5f_train = h5py.File(str(Path(train_path, 'train.h5')), 'r')
        train_char = h5f_train['xTrain_c']

        h5f_test = h5py.File(str(Path(test_path, 'test.h5')), 'r')
        test_char = h5f_test['xTest_c']

        # eval_x, eval_y
        eval_x, eval_y = self.xTest, self.yTest

        # Data generators
        training_generator = DataGenerator([i for i in range(len(self.xTrain))],
                                        x=self.xTrain,
                                        x_flair=train_flair,
                                        x_char=train_char,
                                        y=self.yTrain,
                                        batch_size=self.config.batch_size)
    
        validation_generator = DataGenerator([i for i in range(len(self.xTest))],
                                            x=self.xTest,
                                            x_flair=test_flair,
                                            x_char=test_char,
                                            y=self.yTest,
                                            batch_size=self.config.batch_size)
        
        predict_generator = DataGenerator([i for i in range(len(self.xTest))],
                                        x=self.xTest,
                                        x_flair=test_flair,
                                        x_char=test_char,
                                        y=self.yTest,
                                        batch_size=self.config.batch_size,
                                        pred=True)
    
        return eval_x, eval_y, training_generator, validation_generator, predict_generator


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self, data, x, x_g, y_true_idx, ap):
        self.pre = []
        self.rec = []
        self.f1 = []
        self.best_f1 = 0.
        self.data = data
        self.x = x
        self.x_g = x_g
        self.y_true_idx = y_true_idx
        self.ap = ap
    
    def on_epoch_end(self, epoch, logs={}):
        'Called at the end of an epoch during training.'
  
        y_pred = np.argmax(self.model.predict_generator(self.x_g), axis=-1)[:len(self.x)]
        y_pred_idx = [final_result(y_pred[i],
                                [self.data.index2word[w] for w in self.x[i] if w != 0]) 
                    for i in range(len(y_pred))]
        pp = sum([len(i) for i in y_pred_idx if i != 0])
        tp = 0

        for i in range(len(self.y_true_idx)):
            if self.y_true_idx[i] != 0 and y_pred_idx[i] != 0:
                for m in self.y_true_idx[i]:
                    y_true_cause = [self.data.index2word[self.x[i][idx]] for idx in m[0]]
                    y_true_effect = [self.data.index2word[self.x[i][idx]] for idx in m[-1]]

                    for n in y_pred_idx[i]:
                        y_pred_cause = [self.data.index2word[self.x[i][idx]] for idx in n[0] 
                                if self.x[i][idx] != 0]
                        y_pred_effect = [self.data.index2word[self.x[i][idx]] for idx in n[-1]
                                if self.x[i][idx] != 0]
                        if y_true_cause == y_pred_cause and y_true_effect == y_pred_effect:
                            tp += 1

        pre = tp / float(pp) if pp != 0 else 0
        rec = tp / float(self.ap) if self.ap != 0 else 0
        f1 = 2 * pre * rec / float(pre + rec) if (pre + rec) != 0 else 0
        
        logs['precision'] = pre
        logs['recall'] = rec 
        logs['f1'] = f1

        self.pre.append(pre)
        self.rec.append(rec)
        self.f1.append(f1)
    
        if f1 > self.best_f1:
            self.best_f1 = f1

        # Log metrics using wandb.log
        wandb.log({'precision': pre,
                    'recall': rec, 
                    'f1': f1,
                    'best_f1': self.best_f1})

        print(' - val_precision: %.4f - val_recall: %.4f - val_f1_score: %.4f - best_f1_score: %.4f' %
            (pre, rec, f1, self.best_f1))


class CausalityExtractor:
    def __init__(self, config):
        self.config = config
        self.reproducibility()
        self.kernel_initializer = keras.initializers.glorot_uniform(seed=config.seed)
        self.recurrent_initializer = keras.initializers.Orthogonal(seed=config.seed)
        self.lr = config.learning_rate
        self.save_path = save_path
    
    def reproducibility(self):
        """
        Ensure that the model can obtain reproducible results
        """
        os.environ['PYTHONHASHSEED'] = str(self.config.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.cuda_devices)
        np.random.seed(self.config.seed)
        rn.seed(self.config.seed)
        session_conf =  tf.compat.v1.ConfigProto(                    
                                    device_count = {'CPU': self.config.cpu_core},
                                    intra_op_parallelism_threads = self.config.cpu_core,
                                    inter_op_parallelism_threads = self.config.cpu_core,
                                    gpu_options =  tf.compat.v1.GPUOptions(allow_growth=True  
                                          #per_process_gpu_memory_fraction=0.09
                                                                ),
                                    allow_soft_placement=True)
    
        tf.compat.v1.set_random_seed(self.config.seed) # tf.random.set_seed(config.seed) 
        sess =  tf.compat.v1.Session(graph= tf.compat.v1.get_default_graph(), config=session_conf)
        K.set_session(sess)

    def slm(self, data):
        """
        Returns Sequence Labeling Model.
        """
        # Extvec embedding
        seq = Input(shape=(None,), name='INPUT')
        emb = Embedding(data.VOCAB_SIZE, EXTVEC_DIM, weights=[data.embedding], 
                        mask_zero=True, trainable=False, name='WE')(seq)
        input_node = [seq]

        # Flair embedding
        flair = Input(shape=(None, FLAIR_DIM), name='FLAIR')
        emb = concatenate([emb, flair], axis=-1, name='EMB_FLAIR')
        input_node.append(flair)

        # Char embedding
        char_seq = Input(shape=(None, None), name='CHAR_INPUT')
        char_embedding = []
    
        for _ in range(CHAR_SIZE):
            scale = math.sqrt(3.0 / CHAR_DIM)
            char_embedding.append(np.random.uniform(-scale, scale, CHAR_DIM))
        char_embedding = np.asarray(char_embedding)

        char_emb = TimeDistributed(Embedding(CHAR_SIZE, CHAR_DIM, 
                                            weights=[char_embedding],
                                            mask_zero=True, trainable=True), 
                                    name='CHAR_EMB')(char_seq)
    
        # Char_emb: CNN 
        char_emb = TimeDistributed(MaskConv1D(filters=NUM_CHAR_CNN_FILTER,
                                            kernel_size=CHAR_CNN_KERNEL_SIZE,
                                            padding='same',
                                            kernel_initializer=self.kernel_initializer),
                                    name='CHAR_CNN')(char_emb)
        char_emb = TimeDistributed(
                    Lambda(lambda x: K.max(x, axis=1)), name='MAX_POOLING')(char_emb)
      
        input_node.append(char_seq)
        emb = concatenate([emb, char_emb], axis=-1, name='EMB_CHAR') 

        # Backbone: LSTM
        dec = Bidirectional(LSTM(self.config.lstm_size,
                                kernel_initializer = self.kernel_initializer,
                                recurrent_initializer = self.recurrent_initializer,
                                dropout = self.config.dropout_rate,
                                recurrent_dropout = self.config.dropout_rate,
                                implementation=2,
                                return_sequences=True),
                merge_mode='concat', name='BiLSTM-1')(emb)
        
        # Multihead self attention
        mhsa = MultiHeadSelfAttention(head_num = self.config.nb_head,
                                    size_per_head = self.config.size_per_head,
                                    kernel_initializer = self.kernel_initializer,
                                    name='MHSA')(dec)
        dec = concatenate([dec, mhsa], axis=-1, name='CONTEXT')

        # Classifier: crf 
        dense = TimeDistributed(Dense(NUM_CLASS, activation=None, 
                                    kernel_initializer = self.kernel_initializer),
                                name='DENSE')(dec)
        crf = ChainCRF(init = self.kernel_initializer, name='CRF')
        output = crf(dense)

        loss_func = crf.sparse_loss
        optimizer = optimizers.Nadam(lr = self.lr, clipnorm = self.config.clip_norm)

        model = Model(inputs=input_node, outputs=output)
        model.compile(loss=loss_func, optimizer=optimizer)

        return model

    def cv(self, data_cv):
        """Cross validation"""
        model = self.slm(data_cv)
        model.summary()

        eval_x, eval_y, training_generator, validation_generator, predict_generator = data_cv.cross_validation()

        y_true = eval_y.reshape(eval_y.shape[0], MAX_WLEN)
        y_true_idx = [final_result(y_true[i], 
                                    [data_cv.index2word[w] for w in eval_x[i] if w != 0])
                    for i in range(len(y_true))]
        ap = sum([len(i) for i in y_true_idx if i != 0])
    
        # Callbacks
        wb = WandbCallback()

        evaluator = Evaluate(data_cv, eval_x, predict_generator, y_true_idx, ap)
                         
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                    verbose=1, cooldown=5, min_lr=0.00005) 
    
        es = EarlyStopping(monitor='f1', mode='max', patience=20, verbose=1)
    
        model.fit_generator(training_generator, epochs = self.config.num_epochs, 
                            verbose = 1,
                            validation_data = validation_generator,
                            callbacks=[wb, evaluator, reduce_lr, es],
                            shuffle=False)
    
        # Log metrics using wandb.log
        wandb.log({'k_fold': self.config.k_fold})
    
    def train(self, data_train):
        """Train"""
        model = self.slm(data_train)
        model.summary()

        eval_x, eval_y, training_generator, validation_generator, predict_generator = data_train.train()
    
        y_true = eval_y.reshape(eval_y.shape[0], MAX_WLEN)
        y_true_idx = [final_result(y_true[i], 
                                [data_train.index2word[w] for w in eval_x[i] if w != 0])
                    for i in range(len(y_true))]
        ap = sum([len(i) for i in y_true_idx if i != 0])
    
        # Callbacks
        wb = WandbCallback()

        evaluator = Evaluate(data_train, eval_x, predict_generator, y_true_idx, ap)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                    verbose=1, cooldown=5, min_lr=0.00005)
    
        es = EarlyStopping(monitor='f1', mode='max', patience=20, verbose=1)

        fp3 = str(Path(save_path, 'best_model.h5'))
        mc = ModelCheckpoint(fp3, monitor='f1', mode='max', 
                            verbose=1, save_best_only=True, save_weights_only=True)
    
        # Fit model
        model.fit_generator(training_generator, epochs = self.config.num_epochs, 
                            verbose=1, validation_data=validation_generator, 
                            callbacks=[wb, evaluator, reduce_lr, es, mc],
                            shuffle=False)