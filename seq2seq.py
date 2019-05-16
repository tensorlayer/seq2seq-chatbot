#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
from cross_entropy_seq_loss import *
from tqdm import tqdm
from sklearn.utils import shuffle
from data.twitter import data


class Seq2seq(Model):
    def __init__(
            self,
            batch_size,
            cell_dec,
            cell_enc,
            embedding_layer=None,
            is_train=True,
            name="seq2seq"
    ):
        super(Seq2seq, self).__init__(name=name)
        self.embedding_layer = embedding_layer
        self.vocabulary_size = embedding_layer.vocabulary_size
        self.embedding_size = embedding_layer.embedding_size
        self.encoding_layer = tl.layers.RNN(cell=cell_enc, in_channels=self.embedding_size, return_state=True)
        self.decoding_layer = tl.layers.RNN(cell=cell_dec, in_channels=self.embedding_size)
        self.reshape_layer = tl.layers.Reshape([-1, cell_dec.units])
        self.dense_layer = tl.layers.Dense(n_units=self.vocabulary_size, act=tf.nn.softmax, in_channels=cell_dec.units)
        self.reshape_layer_after = tl.layers.Reshape([batch_size, -1, self.vocabulary_size])
        self.reshape_layer_individual_sequence = tl.layers.Reshape([-1, 1, self.vocabulary_size])
        

    def inference(self, encoding, seq_length, start_token):

        # after embedding the encoding sequence, start the encoding_RNN, then transfer the state to decoing_RNN
        after_embedding_encoding = self.embedding_layer(encoding)

        enc_rnn_ouput, state = self.encoding_layer(after_embedding_encoding, return_state=True)

        
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

        feed_output, state = self.decoding_layer(after_embedding_decoding, initial_state=state, return_state=True)
        feed_output = self.reshape_layer(feed_output)
        feed_output = self.dense_layer(feed_output)
        feed_output = self.reshape_layer_individual_sequence(feed_output)
        feed_output = tf.argmax(feed_output, -1)
        
        final_output = feed_output
        
        for i in range(seq_length-1):
            feed_output = self.embedding_layer(feed_output)
            feed_output, state = self.decoding_layer(feed_output, state, return_state=True)
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
            enc_rnn_output, state = self.encoding_layer(after_embedding_encoding, return_state=True)
            
            dec_rnn_output, state = self.decoding_layer(after_embedding_decoding, initial_state=state, return_state=True)
            dec_output = self.reshape_layer(dec_rnn_output)
            denser_output = self.dense_layer(dec_output)

            ## reshape into [Batch_size, seq_step, vocabulary_size]
            output = self.reshape_layer_after(denser_output)
        else:
            encoding = inputs
            output, state = self.inference(encoding, seq_length, start_token)

        if (return_state):
            return output, state
        else:
            return output

def initial_setup(data_corpus):
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    validX = tl.prepro.remove_pad_sequences(validX.tolist())
    validY = tl.prepro.remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY



if __name__ == "__main__":
    batch_size = 32
    data_corpus = "twitter"
    #data preprocessing
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    n_step = src_len // batch_size
    src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    emb_dim = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word

    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    """ A data for Seq2Seq should look like this:
    input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
    decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
    target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
    target_mask : [1, 1, 1, 1, 0]
    """

    num_epochs = 5
    vocabulary_size = src_vocab_size
    batch_size = 32

    # seq_step_input = 3  
    # encoding = np.random.randint(low=0,high=vocabulary_size,size=(batch_size,seq_step_input))
    # decoding = np.random.randint(low=0,high=vocabulary_size,size=(batch_size,seq_step_output))
    # target = np.random.randint(low=0,high=10,size=(batch_size,seq_step_output))

    def inference(seed):
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[seed_id], seq_length=8, start_token=start_id)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w_id == end_id:
                break
            sentence = sentence + [w]
        return sentence

    
    model_ = Seq2seq(
        batch_size = batch_size,
        cell_enc=tf.keras.layers.SimpleRNNCell(units=3),
        cell_dec=tf.keras.layers.SimpleRNNCell(units=3), 
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
        )
    
    optimizer = tf.optimizers.Adam(learning_rate=0.005)
    model_.train()

    seeds = ["happy birthday have a nice day",
                 "donald trump won last nights presidential debate according to snap online polls"]
    for epoch in range(num_epochs):
        model_.train()
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)
            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)

            with tf.GradientTape() as tape:
                ## compute outputs
                output = model_(inputs = [X, _decode_seqs])
                
                output = tf.reshape(output, [-1, vocabulary_size])
                ## compute loss and update model
                loss = cross_entropy_seq(output, _target_seqs)

                grad = tape.gradient(loss, model_.weights)
                optimizer.apply_gradients(zip(grad, model_.weights))

            total_loss += loss
            n_iter += 1
        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))


        print("================= ========= ========== ========== ============ =========== ========= ======= \n\n")
        # inference after every epoch
        for seed in seeds:
            print("Query >", seed)
            sentence = inference(seed)
            print(" >", ' '.join(sentence))
    
    
