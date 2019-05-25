#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer


class Seq2seq_(Model):
    def __init__(
            self,
            decoder_seq_length,
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
        self.n_layer = n_layer
        self.enc_layers = []
        self.dec_layers = []
        # Could we modify it as a list of layers with self-designed n_layers in the building stage?
        for i in range(n_layer):
            if (i == 0):
                self.enc_layers.append(tl.layers.RNN(cell=cell_enc(units=n_units), in_channels=self.embedding_size, return_last_state=True))
            else:
                self.enc_layers.append(tl.layers.RNN(cell=cell_enc(units=n_units), in_channels=n_units, return_last_state=True))

        for i in range(n_layer):
            if (i == 0):
                self.dec_layers.append(tl.layers.RNN(cell=cell_dec(units=n_units), in_channels=self.embedding_size, return_last_state=True))
            else:
                self.dec_layers.append(tl.layers.RNN(cell=cell_dec(units=n_units), in_channels=n_units, return_last_state=True))



        self.reshape_layer = tl.layers.Reshape([-1, n_units])
        self.dense_layer = tl.layers.Dense(n_units=self.vocabulary_size, in_channels=n_units)
        self.reshape_layer_after = tl.layers.Reshape([-1, decoder_seq_length, self.vocabulary_size])
        self.reshape_layer_individual_sequence = tl.layers.Reshape([-1, 1, self.vocabulary_size])

    def inference(self, encoding, seq_length, start_token, top_n):

        # after embedding the encoding sequence, start the encoding_RNN, then transfer the state to decoing_RNN
        feed_output = self.embedding_layer(encoding)

        state = [None for i in range(self.n_layer)]

        for i in range(self.n_layer):
            feed_output, state[i] = self.enc_layers[i](feed_output, return_state=True)

        batch_size = len(encoding) 
        decoding = [[start_token] for i in range(batch_size)]
        feed_output = self.embedding_layer(decoding)

        for i in range(self.n_layer):
            feed_output, state[i] = self.dec_layers[i](feed_output, initial_state=state[i], return_state=True)
        
        feed_output = self.reshape_layer(feed_output)
        feed_output = self.dense_layer(feed_output)
        feed_output = self.reshape_layer_individual_sequence(feed_output)

        if (top_n is not None):
            idx = np.argpartition(feed_output[0][0], -top_n)[-top_n:]
            probs = [feed_output[0][0][i] for i in idx]
            probs = probs / np.sum(probs)
            feed_output = np.random.choice(idx, p=probs)
            feed_output = tf.convert_to_tensor([[feed_output]])
        else:
            feed_output = tf.argmax(feed_output, -1)
        final_output = feed_output
        for i in range(seq_length - 1):
            feed_output = self.embedding_layer(feed_output)
            for i in range(self.n_layer):
                feed_output, state[i] = self.dec_layers[i](feed_output, initial_state=state[i], return_state=True)
            feed_output = self.reshape_layer(feed_output)
            feed_output = self.dense_layer(feed_output)
            feed_output = self.reshape_layer_individual_sequence(feed_output)

            if (top_n is not None):
                idx = np.argpartition(feed_output[0][0], -top_n)[-top_n:]
                probs = [feed_output[0][0][i] for i in idx]
                probs = probs / np.sum(probs)
                feed_output = np.random.choice(idx, p=probs)
                feed_output = [[feed_output]]
            else:
                feed_output = tf.argmax(feed_output, -1)
            final_output = tf.concat([final_output, feed_output], 1)

        return final_output, state

    def forward(self,
                inputs,
                seq_length=8,
                start_token=None,
                return_state=False,
                top_n = None):

        state = [None for i in range(self.n_layer)]
        if (self.is_train):
            encoding = inputs[0]
            enc_output = self.embedding_layer(encoding)


            for i in range(self.n_layer):
                enc_output, state[i] = self.enc_layers[i](enc_output, return_state=True)

            decoding = inputs[1]
            dec_output = self.embedding_layer(decoding)

            for i in range(self.n_layer):
                dec_output, state[i] = self.dec_layers[i](dec_output, initial_state=state[i], return_state=True)

            dec_output = self.reshape_layer(dec_output)
            denser_output = self.dense_layer(dec_output)
            output = self.reshape_layer_after(denser_output)
        else:
            encoding = inputs
            output, state = self.inference(encoding, seq_length, start_token, top_n)

        if (return_state):
            return output, state
        else:
            return output
