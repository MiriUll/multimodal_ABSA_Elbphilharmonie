import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Bidirectional, Attention, Dense, Activation, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import time
import json
from sklearn.model_selection import train_test_split

print('Loading embedding..')
embeddings_dict = {}
dim = 100
f = open('../datastories-semeval2017-task4/embeddings/datastories.twitter.100d.txt', "r", encoding="utf-8")
for i, line in enumerate(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_dict[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_dict))

def get_embeddings(vectors, dim):
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    wv_map = {}
    pos = 0
    # +1 for zero padding token and +1 for unk
    emb_matrix = np.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        pos = i + 1
        wv_map[word] = pos
        emb_matrix[pos] = vector

    # add unknown token
    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = np.random.uniform(low=-0.05, high=0.05, size=dim)
    
    return emb_matrix, wv_map
embeddings, word_indices = get_embeddings(embeddings_dict, dim)

print('Loading data...')
data = pd.read_csv('data/SemEval2017/train_data.csv')
data['preprocessed'] = data.preprocessed.apply(lambda x: ast.literal_eval(x))
data_test = pd.read_csv('data/SemEval2017/test_data.csv')
data_test['preprocessed'] = data_test.preprocessed.apply(lambda x: ast.literal_eval(x))
X_train_vector = np.loadtxt('data/SemEval2017/train_data_vectorized.csv')
X_test_vector = np.loadtxt('data/SemEval2017/test_data_vectorized.csv')

sentiment_id = {'positive': 0, 'neutral': 1, 'negative': 2}
y_train = to_categorical(data.sentiment.apply(lambda x: sentiment_id[x]))
y_test = to_categorical(data_test.sentiment.apply(lambda x: sentiment_id[x]))
max_length = max(data.preprocessed.apply(lambda x: len(x)))
def padded_index_vector(df):
    X = df.preprocessed.apply(lambda x: [word_indices[word] if word in word_indices else word_indices['<unk>'] for word in x]).values
    X = [np.pad(np.array(x), (0, max_length -len(x)), 'constant', constant_values=(0,0)) for x in X]
    X = np.stack(X)
    return X
X_train = padded_index_vector(data)
X_test = padded_index_vector(data_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)


print('Loaded data, start building model')

# https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
class Attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
    
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                           initializer="glorot_uniform", trainable=True)
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                           initializer="glorot_uniform", trainable=True)
    
        super(Attention, self).build(input_shape)

    def call(self, x):
    
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
    
#         if self.return_sequences:
#             return a, output
    
#         return a, tf.keras.backend.sum(output, axis=1)
        if self.return_sequences:
            return output
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences 
        })
        return config


def embeddings_layer(max_length, embeddings, samples, trainable=False, masking=False,
                     scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        #input_shape=(max_length),
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding
emb_layer = embeddings_layer(max_length, embeddings, len(data))

model = Sequential()
model.add(emb_layer)
model.add(Dropout(0.3))
layers = 2
for i in range(layers):
    rs = (layers > 1 and i < layers - 1)
    rnn = LSTM(64, return_sequences=True, dropout=0.3)
    model.add(Bidirectional(rnn))
    model.add(Dropout(0.3))
#model.add(Attention(name='attention_weight'))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(len(data.sentiment.unique()), activity_regularizer=L2(0.0001)))
model.add(Activation('softmax'))

model.compile(optimizer=Adam(clipnorm=1, lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

print('Initial evaluation')
print(model.evaluate(X_val, y_val))

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), validation_freq=5, epochs=50,
                    batch_size=50, use_multiprocessing=True, workers=20, shuffle=True)

history = history.history
timestamp = str(int(time.time()))
json.dump(history, open('training_history_' + timestamp + '.json', 'w'))
plt.subplot(2, 1, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='upper right')
plt.ylabel('CCE')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('training_' + timestamp + '.png')

print('Saving model')
model.save('model_' + timestamp + '.h5', save_format='h5')
model.save('model_' + timestamp)


print('Saving model')
model.save('model_' + timestamp + '.h5', save_format='h5')
model.save('model_' + timestamp)

