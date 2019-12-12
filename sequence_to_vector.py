# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build your own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError



class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self.input_dim = input_dim
        self.num_layers= num_layers
        self.GRU= tf.keras.layers.GRU(self.input_dim, activation= 'tanh', return_sequences=True, return_state= True)
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        self.sequence_mask= sequence_mask

        layer_representations= [] #define en empty list for layer_reps
        GRU1= self.GRU(vector_sequence, mask=self.sequence_mask) 
        layer_representations.append(GRU1[1]) #add GRU1 to layer_reps
        for i in range(2,self.num_layers+1): 
            newGRU= self.GRU(GRU1, mask=self.sequence_mask) #get the last GRU and create a new GRU                                      
            GRU1= newGRU #set the last GRU to be this new GRU 
            layer_representations.append(newGRU[1]) #add the last GRU layer to layer_reps
        combined_vector=newGRU[1] #get the last dimension of GRU for combined_vector

        layer_representations = tf.convert_to_tensor(layer_representations, dtype=tf.float32) #convert layer_reps to tensor
        layer_representations = tf.transpose(layer_representations, perm= [1,0,2]) #correct the dimensions of layer_reps
        # TODO(students): end
        word_representations=[]
 #       import pdb; pdb.set_trace()
        for i in range(vector_sequence.shape[1]):
          word_representations.append(newGRU[0][:,i,:])
        #import pdb; pdb.set_trace()
        word_representations= tf.convert_to_tensor(word_representations, dtype=tf.float32)
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations,
                "word_representations": word_representations}
