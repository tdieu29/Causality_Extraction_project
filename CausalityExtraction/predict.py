from pathlib import Path
import os
import numpy as np
import pickle
import h5py
import wandb
import argparse
from argparse import Namespace

from CausalityExtraction import configurations, model
from CausalityExtraction.model import MaskConv1D, DataGenerator, CausalityExtractor
from tag2triplet import *


#wandb.login()

# Constants and arguments
index_path = configurations.INDEX_DIR
embedding_path = configurations.EMBEDDING_DIR
predict_path = configurations.PREDICT_DIR
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

config_dict = Namespace(
    cuda_devices = 0,
    cpu_core = 1,
    seed = 2,
    k_fold = 0,
    lstm_size = 256,
    dropout_rate = 0.5,
    nb_head = 3,
    size_per_head = 8,
    learning_rate = 0.001,
    clip_norm = 5.0,
    num_epochs = 200,
    batch_size = 16
)

#run = wandb.init(project = 'predict', config = config_dict)
#config = wandb.config

class Data:
    def __init__(self):
        fp1 = str(Path(index_path, 'index_w.pkl'))
        self.word2index, self.index2word = pickle.load(open(fp1, 'rb'))

        fp2 = str(Path(embedding_path, 'extvec_embedding.npy'))
        self.embedding = np.load(open(fp2, 'rb'))
      
        self.VOCAB_SIZE = len(self.word2index) 

        fp3 = str(Path(predict_path, 'input.h5'))
        with h5py.File(fp3, 'r') as fh:
            self.inputWordArray = fh['inputWordArray'][:]
            self.inputCharArray = fh['inputCharArray'][:]

        fp4 = str(Path(embedding_path, 'input_flair.h5'))
        with h5py.File(fp4, 'r') as f:
            self.input_flair = f['input_flair'][:]


def predict(config = config_dict):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    data = Data()
    extractor = CausalityExtractor(config_dict)
    model = extractor.slm(data)

    model.load_weights(Path(save_path, 'best_model.h5'))

    test_generator = DataGenerator([i for i in range(len(data.inputWordArray))],
                                    x = data.inputWordArray,
                                    x_flair = data.input_flair,
                                    x_char = data.inputCharArray,
                                    y = np.zeros((len(data.inputWordArray), MAX_WLEN, 1)),
                                    batch_size=config.batch_size,
                                    pred=True)

    result = model.predict_generator(test_generator)
    prediction = np.argmax(result, axis=-1)[:len(data.inputWordArray)]

    #Print predict causal triplets 
    decoded_predictions = {
        'Sentences': [],
        'Predictions': []
    }   

    flag = 1
    for i, p in enumerate(prediction):
        p_idx = final_result(p, [data.index2word[w] for w in data.inputWordArray[i] if w != 0])

        decoded_predictions['Sentences'].append('Sentence-%.3d:' % flag + \
                                     ' '.join([data.index2word[w] for w in data.inputWordArray[i] if w != 0]))
        
        if p_idx != 0:
            y_pred_Cause = [[data.index2word[data.inputWordArray[i][idx]] 
                            for idx in n[0] if data.inputWordArray[i][idx] != 0 ]
                            for n in p_idx]

            y_pred_Effect = [[data.index2word[data.inputWordArray[i][idx]] 
                            for idx in n[-1] if data.inputWordArray[i][idx] != 0]
                            for n in p_idx]

            decoded_predictions['Predictions'].append([(y_pred_Cause[i], y_pred_Effect[i]) for i in range(len(y_pred_Cause))])
        else:
            decoded_predictions['Predictions'].append([])
        
        flag += 1

    return decoded_predictions
