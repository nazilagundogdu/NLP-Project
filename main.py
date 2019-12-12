# inbuilt lib imports:
from typing import List, Dict, Tuple
import os

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models

# project imports
from sequence_to_vector import GruSequenceToVector


class MainClassifier(models.Model):
    def __init__(self,
                 seq2vec_choice: str,
                 vocab_size: int,
                 embedding_dim: int,
                 num_layers: int = 2,
                 num_classes = 2) -> 'MainClassifier':
        """

        Parameters
        ----------
        seq2vec_choice : ``str``
        vocab_size : ``int``
            Vocabulary size used to index the data instances.
        embedding_dim : ``int``
            Embedding matrix dimension
        num_layers : ``int``
            Number of layers of sentence encoder to build.
        num_classes : ``int``
            Number of classes that this Classifier chooses from.
        """
        super(MainClassifier, self).__init__()
        # Construct and setup sequence_to_vector model

       
        self._seq2vec_layer = GruSequenceToVector(embedding_dim, num_layers)

        # Trainable Variables
        self._embeddings = tf.Variable(tf.random.normal((vocab_size, embedding_dim)),
                                       trainable=True)
        self._classification_layer = layers.Dense(units=num_classes)

    def call(self,
             inputs: tf.Tensor,
             training=False):
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        inputs : ``str``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, max_tokens_num) and entries are indices of tokens
            in to the vocabulary. 0 means that it's a padding token. max_tokens_num
            is maximum number of tokens in any text sequence in this batch.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.
        """
        embedded_tokens = tf.nn.embedding_lookup(self._embeddings, inputs)
        tokens_mask = tf.cast(inputs!=0, tf.float32)
        outputs = self._seq2vec_layer(embedded_tokens, tokens_mask, training)
        sentence_vector = outputs["combined_vector"]
        layer_representations = outputs["layer_representations"]
        word_representations= outputs["word_representations"]
        final = []
        x = len(word_representations)
        for index in range(x):
            a = tf.stack([word_representations[index] for x in range(x)],1)
            b = tf.stack([sentence_vector for x in range(x)],1)
            c = tf.stack(word_representations,1)
            
            final.append(tf.concat([a,b,c],2))
        
        final = tf.stack(final,1)

        
        logits = self._classification_layer(final)
        return {"logits": logits}
