#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Seq2seq_(Model):
    def __init__(
            self,
            batch_size,
            cell_enc,
            cell_dec,
            n_units=256,
            n_layer=3,
            embedding_layer=None,
            is_train=True,
            name="seq2seq_"
    ):
        super(Seq2seq_, self).__init__(name=name)
        self.embedding_layer = embedding_layer
        self.vocabulary_size = embedding_layer.vocabulary_size
        self.embedding_size = embedding_layer.embedding_size

        # Could we modify it as a list of layers with self-designed n_layers in the building stage?
        self.encoding_layer_0 = tl.layers.RNN(cell=cell_enc(units=n_units), in_channels=self.embedding_size, return_last_state=True)
        self.encoding_layer_1 = tl.layers.RNN(cell=cell_enc(units=n_units), in_channels=n_units, return_last_state=True)
        self.encoding_layer_2 = tl.layers.RNN(cell=cell_enc(units=n_units), in_channels=n_units, return_last_state=True)


        self.decoding_layer_0 = tl.layers.RNN(cell=cell_dec(units=n_units), in_channels=self.embedding_size, return_last_state=True)
        self.decoding_layer_1 = tl.layers.RNN(cell=cell_dec(units=n_units), in_channels=n_units, return_last_state=True)
        self.decoding_layer_2 = tl.layers.RNN(cell=cell_dec(units=n_units), in_channels=n_units, return_last_state=True)


        self.reshape_layer = tl.layers.Reshape([-1, n_units])
        self.dense_layer = tl.layers.Dense(n_units=self.vocabulary_size, in_channels=n_units)
        self.reshape_layer_after = tl.layers.Reshape([batch_size, -1, self.vocabulary_size])
        self.reshape_layer_individual_sequence = tl.layers.Reshape([-1, 1, self.vocabulary_size])
        

    def inference(self, encoding, seq_length, start_token):

        # after embedding the encoding sequence, start the encoding_RNN, then transfer the state to decoing_RNN
        after_embedding_encoding = self.embedding_layer(encoding)

        enc_rnn_output, state = self.encoding_layer_0(after_embedding_encoding, return_state=True)
        enc_rnn_output, state = self.encoding_layer_1(enc_rnn_output, return_state=True)
        enc_rnn_output, state = self.encoding_layer_2(enc_rnn_output, return_state=True)

        
        # for the start_token, first create a batch of it, get[Batchsize, 1]. 
        # then embbeding, get[Batchsize, 1, embeddingsize]
        # then RNN, get[Batchsize, 1, RNN_units]
        # then reshape, get[Batchsize*1, RNN_units]
        # then dense, get[Batchsize*1, vocabulary_size]
        # then reshape, get[Batchsize, 1, vocabulary_size]
        # finally, get Argmax of the last dimension, get next_sequence[Batchsize, 1]
        # this next_sequence will repeat above procedure for the sequence_length time

        
        batch_size = len(encoding)
        decoding = [[start_token] for i in range(batch_size)]
        decoding = np.array(decoding)
        
        after_embedding_decoding = self.embedding_layer(decoding)

        feed_output, state = self.decoding_layer_0(after_embedding_decoding, initial_state=state, return_state=True)
        feed_output, state = self.decoding_layer_1(feed_output, initial_state=state, return_state=True)
        feed_output, state = self.decoding_layer_2(feed_output, initial_state=state, return_state=True)

        feed_output = self.reshape_layer(feed_output)
        feed_output = self.dense_layer(feed_output)
        feed_output = self.reshape_layer_individual_sequence(feed_output)
        feed_output = tf.argmax(feed_output, -1)
        
        final_output = feed_output
        
        for i in range(seq_length-1):
            feed_output = self.embedding_layer(feed_output)
            feed_output, state = self.decoding_layer_0(feed_output, initial_state=state, return_state=True)
            feed_output, state = self.decoding_layer_1(feed_output, initial_state=state, return_state=True)
            feed_output, state = self.decoding_layer_2(feed_output, initial_state=state, return_state=True)
            feed_output = self.reshape_layer(feed_output)
            feed_output = self.dense_layer(feed_output)
            feed_output = self.reshape_layer_individual_sequence(feed_output)
            feed_output = tf.argmax(feed_output, -1)
            final_output = tf.concat([final_output,feed_output], 1)


        return final_output, state


    def forward(self, inputs, seq_length=8, start_token=None, return_state=False):


        if (self.is_train):
            encoding = inputs[0]
            after_embedding_encoding = self.embedding_layer(encoding)
            decoding = inputs[1]
            after_embedding_decoding = self.embedding_layer(decoding)

            enc_rnn_output, state = self.encoding_layer_0(after_embedding_encoding, return_state=True)
            enc_rnn_output, state = self.encoding_layer_1(enc_rnn_output, return_state=True)
            enc_rnn_output, state = self.encoding_layer_2(enc_rnn_output, return_state=True)



            decoding = inputs[1]
            after_embedding_decoding = self.embedding_layer(decoding)
            dec_rnn_output, state = self.decoding_layer_0(after_embedding_decoding, initial_state=state, return_state=True)
            dec_rnn_output, state = self.decoding_layer_1(dec_rnn_output, initial_state=state, return_state=True)
            dec_rnn_output, state = self.decoding_layer_2(dec_rnn_output, initial_state=state, return_state=True)

            dec_output = self.reshape_layer(dec_rnn_output)
            denser_output = self.dense_layer(dec_output)
            output = self.reshape_layer_after(denser_output)
        else:
            encoding = inputs
            output, state = self.inference(encoding, seq_length, start_token)

        if (return_state):
            return output, state
        else:
            return output