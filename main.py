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
    data_corpus = "twitter"
    #data preprocessing
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    batch_size = 32
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

    num_epochs = 100
    vocabulary_size = src_vocab_size
    


    def inference(seed):
        model_.eval()
        seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]
        sentence_id = model_(inputs=[seed_id], seq_length=30, start_token=start_id)
        sentence = []
        for w_id in sentence_id[0]:
            w = idx2word[w_id]
            if w == 'end_id':
                break
            sentence = sentence + [w]
        return sentence

    
    model_ = Seq2seq_(
        batch_size = batch_size,
        cell_enc=tf.keras.layers.GRUCell(units=5),
        cell_dec=tf.keras.layers.GRUCell(units=5), 
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
        )
    
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
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
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            with tf.GradientTape() as tape:
                ## compute outputs
                output = model_(inputs = [X, _decode_seqs])
                
                output = tf.reshape(output, [-1, vocabulary_size])
                ## compute loss and update model
                #print(output, _target_seqs, _target_mask)
                loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

                grad = tape.gradient(loss, model_.all_weights)
                optimizer.apply_gradients(zip(grad, model_.all_weights))
                

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
    
    
