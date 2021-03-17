from pathlib import Path
import pickle
import keras
import numpy as np
import h5py
import flair
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from tqdm import tqdm
from flair.data import Sentence

from CausalityExtraction import configurations


# Constants
MAX_WLEN = 58
MAX_CLEN = 23
FLAIR_DIM = 4096

# Load data
fp1 = str(Path(configurations.INDEX_DIR, 'index_w.pkl'))
word2index, index2word = pickle.load(open(fp1, 'rb'))

#fp1 = 'https://drive.google.com/uc?export=download&id=1RnQ8vutbAPpsPqpgsZ6ktGJCRI_5K_E5'
#urllib.request.urlretrieve(fp1, 'index_w.pkl')
#word2index, index2word = pickle.load(open('index_w.pkl', 'rb'))

fp2 = str(Path(configurations.INDEX_DIR, 'index_c.pkl'))
char2index, index2char = pickle.load(open(fp2, 'rb'))

#fp2 = 'https://drive.google.com/uc?export=download&id=1AP7VYaucrSRy6jIkoBqCnOrBhUR4mNL3'
#urllib.request.urlretrieve(fp2, 'index_c.pkl')
#char2index, index2char = pickle.load(open('index_c.pkl', 'rb'))

def sep_pm(ws, pm):
    """
    Separate punctuation mark: , : ; " 
    """
    if pm in [',', ':', ';', '"']:
        for t in range(2):
            for i in range(len(ws)):
                if ws[i].endswith(pm) and ws[i] != pm:
                    ws[i] = ws[i][:-1]
                    ws.insert(i+1, pm)
                  
    if pm in ['"']:
        for t in range(2):
            for i in range(len(ws)):
                if ws[i].startswith(pm) and ws[i] != pm:
                    ws[i] = ws[i][1:]
                    ws.insert(i, pm)


def wordSeg(s):
    """Word segmentation"""
    ws = text_to_word_sequence(s, filters='!#$&*+.%/<=>?@[\\]^_`{|}~\t\n',
                                lower=False, split=' ')

    for pm in [',', ':', ';', '"']:
        sep_pm(ws, pm)

    return ws


def char_data(charList):
    result = []
    for s in charList:
        result.append(np.concatenate((
                        pad_sequences([[char2index.get(c, 1) for c in w] for w in s], 
                                        maxlen=MAX_CLEN, padding='post', truncating='post'),
                        np.zeros((MAX_WLEN-len(s), MAX_CLEN))), axis=0))
    return np.array(result)


def flair_cse(sw):
    """Convert sentence to contextual string embeddings with flair"""
    charlm_embedding_forward = FlairEmbeddings('news-forward')
    charlm_embedding_backward = FlairEmbeddings('news-backward')
    stacked_embeddings = StackedEmbeddings(
                        embeddings=[charlm_embedding_forward, charlm_embedding_backward])
    result = []
    nsw = [Sentence(' '.join(i)) for i in sw]
    for s in tqdm(nsw):
        stacked_embeddings.embed(s)
    
        result.append(np.concatenate((np.array([np.array((token.embedding).cpu()) for token in s]), 
                                  np.zeros((MAX_WLEN-len(s), FLAIR_DIM))), axis=0))
    return np.array(result)

 
def get_input(sentenceList):

    # Add input sentences into a list
    inputSentences = []
    inputSentences.extend(sentenceList)

    # Segment words and characters
    inputWords = [wordSeg(i) for i in inputSentences]
    inputChars = [[list(w) for w in s] for s in inputWords]

    # Add padding to character array  
    inputCharArray = char_data(inputChars)

    # Add padding to word array
    unk_words_dict = dict()
    inputWordSeq = []
    s_idx = 0
    for s in inputWords:
        unk_words_dict['Words_{}'.format(s_idx)] = []
        unk_words_dict['Index_{}'.format(s_idx)] = []

        w_idx = 0
        inputWordSeq_temp = []
        for w in s:
            index = word2index.get(w, None)
            if index == None:
                index = 1
                unk_words_dict['Words_{}'.format(s_idx)].append(w)
                unk_words_dict['Index_{}'.format(s_idx)].append(w_idx)
            inputWordSeq_temp.append(index)
            w_idx += 1
        inputWordSeq.append(inputWordSeq_temp)
        s_idx += 1

    inputWordArray = pad_sequences(inputWordSeq, maxlen=MAX_WLEN, padding='post', truncating='post')

    # Get flair embedding for input sentences
    input_flair = flair_cse(inputWords)

    # Save inputCharArray, inputWordArray and input_flair
    fp3 = str(Path(configurations.PREDICT_DIR, 'input.h5'))
    h5f = h5py.File(fp3, 'w')
    h5f.create_dataset('inputCharArray', data = inputCharArray)
    h5f.create_dataset('inputWordArray', data = inputWordArray)
    h5f.close()

    fp4 = str(Path(configurations.EMBEDDING_DIR, 'input_flair.h5'))
    h5f = h5py.File(fp4, 'w')
    h5f.create_dataset('input_flair', data = input_flair)
    h5f.close()

    # Save unk_words_dict
    fp5 = str(Path(configurations.PREDICT_DIR, 'unk_words_dict.pkl'))
    with open(fp5, 'wb') as fp:
        pickle.dump(unk_words_dict, fp, -1)

