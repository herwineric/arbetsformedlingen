
from tensorflow import keras
import pickle
from collections import defaultdict
from transformers import BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split


with open("texts_idsPreProcessed.pkl", "rb") as pFile:
    X = pickle.load(pFile) 

with open("tagsPreProcessed.pkl", "rb") as pFile:
    Y = pickle.load(pFile) 


with open("metaData.pkl", "rb") as pFile:
    metaData = pickle.load(pFile) 




print("Loading BERT-model")
pretrained_model_name = 'af-ai-center/bert-base-swedish-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)




maxVocab = metaData["lenVocab"] #Length of vocab
lenTexts_pre = [len(X[key]) for key in list(X.keys())]
lenTexts = max(lenTexts_pre)



def tagToID(tag: str) -> int:
    if tag == "L":
        return 3
    elif tag == "S":
        return 2
    else:
        return 1

keyss = list(Y.keys())

dataList = [X[key] for key in keyss]

paddedDataList = keras.preprocessing.sequence.pad_sequences(maxlen = lenTexts, 
                sequences = dataList, value = tokenizer.pad_token_id)



tagList = [[tagToID(t) for t in Y[l].split()] for l in keyss]

paddedTagList = keras.preprocessing.sequence.pad_sequences(maxlen = lenTexts, 
                sequences = tagList, value = tokenizer.pad_token_id)



x_full = paddedDataList
y_full = [keras.utils.to_categorical(t, num_classes = 4) for t in paddedTagList]



X_train, X_split, y_train, y_split = train_test_split(x_full, y_full, test_size = 0.2)
X_valid, X_tset, y_valid, y_test = train_test_split(X_split, y_split, test_size = 0.6)




inp = keras.Input(shape = (lenTexts,))
emb = keras.layers.Embedding(input_dim = maxVocab + 1, output_dim = 50, input_length = lenTexts)(inp)

layer = keras.layers.LSTM(units = 20, return_sequences = True, recurrent_dropout = 0.1)(emb)

out = keras.layers.TimeDistributed(keras.layers.Dense(4, activation = "softmax"))(layer)
#out = keras.layers.Dense(4, activation = "softmax")(out)

opt = keras.optimizers.Adam(lr = 0.001)
model = keras.models.Model(inp, out)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["acc"])

model.fit(X_train, np.array(y_train), batch_size = 32 epochs = 2, verbose = 1)
