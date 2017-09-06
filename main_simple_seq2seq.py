#! /usr/bin/python
# -*- coding: utf8 -*-

"""Sequence to Sequence Learning for Twitter/Cornell Chatbot.

References
----------
http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/
"""

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

import tensorflow as tf
import numpy as np
import time

## select dataset
from data.twitter import data
metadata, idx_q, idx_a = data.load_data(PATH='data/twitter/')                   # Twitter
# from data.cornell_corpus import data
# metadata, idx_q, idx_a = data.load_data(PATH='data/cornell_corpus/')          # Cornell Moive
(trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)

trainX = trainX.tolist()
trainY = trainY.tolist()
testX = testX.tolist()
testY = testY.tolist()
validX = validX.tolist()
validY = validY.tolist()

trainX = tl.prepro.remove_pad_sequences(trainX)
trainY = tl.prepro.remove_pad_sequences(trainY)
testX = tl.prepro.remove_pad_sequences(testX)
testY = tl.prepro.remove_pad_sequences(testY)
validX = tl.prepro.remove_pad_sequences(validX)
validY = tl.prepro.remove_pad_sequences(validY)

## parameters
xseq_len = len(trainX)#.shape[-1]
yseq_len = len(trainY)#.shape[-1]
assert xseq_len == yseq_len
batch_size = 32
n_step = int(xseq_len/batch_size)
xvocab_size = len(metadata['idx2w']) # 8002 (0~8001)
emb_dim = 1024

w2idx = metadata['w2idx']   # dict  word 2 index
idx2w = metadata['idx2w']   # list index 2 word

unk_id = w2idx['unk']   # 1
pad_id = w2idx['_']     # 0
# print(idx2w[8001])
# exit()
start_id = xvocab_size  # 8002
end_id = xvocab_size+1  # 8003
# print(start_id, end_id)
# exit()
w2idx.update({'start_id': start_id})
w2idx.update({'end_id': end_id})
idx2w = idx2w + ['start_id', 'end_id']
# print(idx2w)
xvocab_size = yvocab_size = xvocab_size + 2

""" A data for Seq2Seq should look like this:
input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
target_mask : [1, 1, 1, 1, 0]
"""

print("encode_seqs", [idx2w[id] for id in trainX[10]])
target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
# target_seqs = tl.prepro.remove_pad_sequences([target_seqs], pad_id=pad_id)[0]
print("target_seqs", [idx2w[id] for id in target_seqs])
# exit()
decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
# decode_seqs = tl.prepro.remove_pad_sequences([decode_seqs], pad_id=pad_id)[0]
print("decode_seqs", [idx2w[id] for id in decode_seqs])
target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
print("target_mask", target_mask)
print(len(target_seqs), len(decode_seqs), len(target_mask))
# exit()

## model
def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = xvocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.contrib.rnn.BasicLSTMCell,
                n_hidden = emb_dim,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')
        net_out = DenseLayer(net_rnn, n_units=xvocab_size, act=tf.identity, name='output')
    return net_out, net_rnn

encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") # tl.prepro.sequences_get_mask()

net_out, _ = model(encode_seqs, decode_seqs, is_train=True, reuse=False)

encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
net, net_rnn = model(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
y = tf.nn.softmax(net.outputs)

# print(net_out.outputs)    # (?, 8004)
# print(target_seqs)    # (32, ?)
    # loss_weights = tf.ones_like(target_seqs, dtype=tf.float32)
    # loss = tf.contrib.legacy_seq2seq.sequence_loss(net_out.outputs, target_seqs, loss_weights, yvocab_size)
loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False, name='cost')

net_out.print_params(False)

lr = 0.0001
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
# Truncated Backpropagation for training
# max_grad_norm = 30
# grads, _ = tf.clip_by_global_norm(tf.gradients(loss, net_out.all_params),max_grad_norm)
# optimizer = tf.train.GradientDescentOptimizer(lr)
# train_op = optimizer.apply_gradients(zip(grads, net_out.all_params))

# sess = tf.InteractiveSession()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name='n.npz', network=net)

## train
n_epoch = 50
for epoch in range(n_epoch):
    epoch_time = time.time()
    ## shuffle training data
    from sklearn.utils import shuffle
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    ## train an epoch
    total_err, n_iter = 0, 0
    for X, Y in tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False):
        step_time = time.time()

        X = tl.prepro.pad_sequences(X)
        _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs)

        _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        ## you can view the data here
        # for i in range(len(X)):
        #     print(i, [idx2w[id] for id in X[i]])
        #     # print(i, [idx2w[id] for id in Y[i]])
        #     print(i, [idx2w[id] for id in _target_seqs[i]])
        #     print(i, [idx2w[id] for id in _decode_seqs[i]])
        #     print(i, _target_mask[i])
        #     print(len(_target_seqs[i]), len(_decode_seqs[i]), len(_target_mask[i]))
        # exit()

        _, err = sess.run([train_op, loss],
                        {encode_seqs: X,
                        decode_seqs: _decode_seqs,
                        target_seqs: _target_seqs,
                        target_mask: _target_mask})

        if n_iter % 200 == 0:
            print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))

        total_err += err; n_iter += 1

        ## inference
        if n_iter % 1000 == 0:
            seeds = ["happy birthday have a nice day",
                    "donald trump won last nights presidential debate according to snap online polls"]
            for seed in seeds:
                print("Query >", seed)
                seed_id = [w2idx[w] for w in seed.split(" ")]
                for _ in range(5):  # 1 Query --> 5 Reply
                    # 1. encode, get state
                    state = sess.run(net_rnn.final_state_encode,
                                    {encode_seqs2: [seed_id]})
                    # 2. decode, feed start_id, get first word
                    #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
                    o, state = sess.run([y, net_rnn.final_state_decode],
                                    {net_rnn.initial_state_decode: state,
                                    decode_seqs2: [[start_id]]})
                    w_id = tl.nlp.sample_top(o[0], top_k=3)
                    w = idx2w[w_id]
                    # 3. decode, feed state iteratively
                    sentence = [w]
                    for _ in range(30): # max sentence length
                        o, state = sess.run([y, net_rnn.final_state_decode],
                                        {net_rnn.initial_state_decode: state,
                                        decode_seqs2: [[w_id]]})
                        w_id = tl.nlp.sample_top(o[0], top_k=2)
                        w = idx2w[w_id]
                        if w_id == end_id:
                            break
                        sentence = sentence + [w]
                    print(" >", ' '.join(sentence))
                    # exit()

    print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epoch, total_err/n_iter, time.time()-epoch_time))

    tl.files.save_npz(net.all_params, name='n.npz', sess=sess)
