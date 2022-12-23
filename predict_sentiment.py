from vectorizer import Datastories_embedding
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

##############################################################
# Change these lines to apply on your custom datasets
##############################################################
message_level = pd.read_csv('scraping/test_dataset_sentiment.csv')
aspect_based = pd.read_csv('scraping/test_dataset_sentiment_aspect.csv')


##############################################################
# End of hardcoded parameters
##############################################################


# This class is used to model the attention mechanism in our models
class Attention(tf.keras.layers.Layer):

    def __init__(self, return_sequences=True, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="glorot_uniform", trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config


def predict_sentiment(model, texts, aspect_based=False, aspects=None, sentiments=None, verbose=True,
                      print_misclassified=False, max_length=1396, max_length_aspect=35):
    """
    Takes a model and posts as input and predicts the sentiment of the posts
    :param model: The model to use for prediction
    :param texts: The posts' texts
    :param aspect_based: True if aspects should be given to the model
    :param aspects: The aspects towards which the sentiment should be predicted. If None, the default aspect
        'Elbphilharmonie in Hamburg' will be used
    :param sentiments: The ground truth sentiment labels. If given, accuracy and f1-score will be printed
    :param print_misclassified: Wether to print the misclassified samples
    :param max_length: Max length of the model's embedding. Leave default values for our model
    :param max_length_aspect: Max embedding length of the aspects. Leave default values for our model
    :return:
    """
    class_to_sentiment = {0: 'positive', 1: 'neutral', 2: 'negative'}
    vecs = []
    for s in texts:
        vecs.append(embedding.index_vector(s, max_length))  # vectorize the text data

    if aspect_based:
        if aspects is not None:
            target_vec = []
            for t in aspects:
                target_vec.append(embedding.index_vector(t, max_length_aspect))  # vectorize the aspect texts
            target_vec = np.stack(target_vec)
        else:
            target_vec = np.tile(embedding.index_vector('Elbphilharmonie in Hamburg', max_length_aspect),
                                 (len(vecs), 1))  # use default aspect |post| times
        preds = model.predict([target_vec, np.stack(vecs)])  # run the prediction with the aspects
    else:
        preds = model.predict(np.stack(vecs))  # run prediction without aspect information

    preds_label = [class_to_sentiment[np.argmax(x).item()] for x in preds]  # map the predictions to sentiment classes
    if verbose:
        print('Predicted:', preds_label)
    if sentiments is not None and verbose:
        print('Correct:', sentiments.values)
        print('Accuracy:', accuracy_score(sentiments, preds_label))
        print('F1 score:', f1_score(sentiments, preds_label, average='macro'))
    if sentiments is not None and print_misclassified:
        for i, (p, s) in enumerate(zip(preds_label, sentiments)):
            if p != s:
                print(p, s, texts[i])
    return preds


embedding = Datastories_embedding()
print("** Message-level prediction")
model_message = load_model('sentiment_models/model_datastories_message_3_class',
                           custom_objects={'Attention': Attention, 'f1_metric': f1_score})
predict_sentiment(model=model_message, texts=message_level.post, aspect_based=False, sentiments=message_level.sentiment,
                  verbose=True, print_misclassified=True)

print("** Aspect-based prediction")
model_aspect = load_model('sentiment_models/model_datastories_target_3_class',
                          custom_objects={'Attention': Attention, 'f1_metric': f1_score})
predict_sentiment(model=model_aspect, texts=aspect_based.post, aspect_based=True, aspects=aspect_based.aspect,
                  sentiments=aspect_based.sentiment, verbose=True)
