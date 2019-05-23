#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from loss import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data.twitter import data
from model_seq2seq import Seq2seq_
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"



if __name__ == "__main__":

    vocabulary_size=20
    emb_dim = 50
    trainX = np.random.randint(20, size=(50,5))
    trainY = np.random.randint(20, size=(50,5))
    trainY[:,0] = 0
    
    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len
    

    num_epochs=1000
    batch_size=32
    n_step = src_len//batch_size

    model_ = Seq2seq_(
        decoder_seq_length = 4,
        cell_enc=tf.keras.layers.GRUCell,
        cell_dec=tf.keras.layers.GRUCell,
        n_layer=3,
        n_units=256,
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
        )
    
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    model_.train()

    
    for epoch in range(num_epochs):
        model_.train()
        trainX, trainY = shuffle(trainX, trainY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

            dec_seq = Y[:,:-1]
            target_seq = Y[:,1:] 

            with tf.GradientTape() as tape:
                ## compute outputs
                output = model_(inputs = [X, dec_seq])
 
                output = tf.reshape(output, [-1, vocabulary_size])

                loss = cross_entropy_seq(logits=output, target_seqs=target_seq)

                grad = tape.gradient(loss, model_.all_weights)
                optimizer.apply_gradients(zip(grad, model_.all_weights))
            
            total_loss += loss
            n_iter += 1
        

        model_.eval()
        test_sample = trainX[0,:].tolist()
        top_n = 1
        for i in range(top_n):
            prediction = model_([test_sample], seq_length = 4, start_token = 0, top_n = top_n)
            print(prediction, trainY[0,1:])
        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))



        
    
    
