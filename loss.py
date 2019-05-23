from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).
  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, default: "sequence_loss_by_example".
  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError(
            "Lengths of logits, weights, and targets must be the same "
            "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                        logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    labels=target, logits=logit)
            else:
                crossent = softmax_loss_function(labels=target, logits=logit)

            # print("crossent = ", crossent)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def cross_entropy_seq(logits, target_seqs,
                      batch_size=None):  # , batch_size=1, num_steps=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for fixed length RNN outputs, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.

    Parameters
    ----------
    logits : Tensor
        2D tensor with shape of `[batch_size * n_steps, n_classes]`.
    target_seqs : Tensor
        The target sequence, 2D tensor `[batch_size, n_steps]`, if the number of step is dynamic, please use ``tl.cost.cross_entropy_seq_with_mask`` instead.
    batch_size : None or int.
        Whether to divide the cost by batch size.
            - If integer, the return cost will be divided by `batch_size`.
            - If None (default), the return cost will not be divided by anything.

    Examples
    --------
    >>> import tensorlayer as tl
    >>> see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.for more details
    >>> input_data = tf.placeholder(tf.int32, [batch_size, n_steps])
    >>> targets = tf.placeholder(tf.int32, [batch_size, n_steps])
    >>> # build the network
    >>> print(net.outputs)
    (batch_size * n_steps, n_classes)
    >>> cost = tl.cost.cross_entropy_seq(network.outputs, targets)

    """
    sequence_loss_by_example_fn = sequence_loss_by_example

    loss = sequence_loss_by_example_fn(
        [logits], [tf.reshape(target_seqs, [-1])],
        [tf.ones_like(tf.reshape(target_seqs, [-1]), dtype=tf.float32)])
    # [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss)  # / batch_size
    if batch_size is not None:
        cost = cost / batch_size
    return cost


def cross_entropy_seq_with_mask(logits,
                                target_seqs,
                                input_mask,
                                return_details=False,
                                name=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for Dynamic RNN with Synced sequence input and output.

    Parameters
    -----------
    logits : Tensor
        2D tensor with shape of [batch_size * ?, n_classes], `?` means dynamic IDs for each example.
        - Can be get from `DynamicRNNLayer` by setting ``return_seq_2d`` to `True`.
    target_seqs : Tensor
        int of tensor, like word ID. [batch_size, ?], `?` means dynamic IDs for each example.
    input_mask : Tensor
        The mask to compute loss, it has the same size with `target_seqs`, normally 0 or 1.
    return_details : boolean
        Whether to return detailed losses.
            - If False (default), only returns the loss.
            - If True, returns the loss, losses, weights and targets (see source code).

    Examples
    --------
    >>> batch_size = 64
    >>> vocab_size = 10000
    >>> embedding_size = 256
    >>> input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input")
    >>> target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target")
    >>> input_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="mask")
    >>> net = tl.layers.EmbeddingInputlayer(
    ...         inputs = input_seqs,
    ...         vocabulary_size = vocab_size,
    ...         embedding_size = embedding_size,
    ...         name = 'seq_embedding')
    >>> net = tl.layers.DynamicRNNLayer(net,
    ...         cell_fn = tf.contrib.rnn.BasicLSTMCell,
    ...         n_hidden = embedding_size,
    ...         dropout = (0.7 if is_train else None),
    ...         sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
    ...         return_seq_2d = True,
    ...         name = 'dynamicrnn')
    >>> print(net.outputs)
    (?, 256)
    >>> net = tl.layers.DenseLayer(net, n_units=vocab_size, name="output")
    >>> print(net.outputs)
    (?, 10000)
    >>> loss = tl.cost.cross_entropy_seq_with_mask(net.outputs, target_seqs, input_mask)

    """
    logits = tf.cast(logits, tf.float64)
    targets = tf.reshape(target_seqs, [-1])  # to one vector
    weights = tf.cast(tf.reshape(input_mask, [-1]),
                      tf.float64)  # to one vector like targets
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets, name=name) * weights
    # print(logits, losses, weights, targets)
    # losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name=name)) # for TF1.0 and others

    loss = tf.divide(
        tf.reduce_sum(
            losses
        ),  # loss from mask. reduce_sum before element-wise mul with mask !!
        tf.reduce_sum(weights),
        name="seq_loss_with_mask")

    if return_details:
        return loss, losses, weights, targets
    else:
        return loss
