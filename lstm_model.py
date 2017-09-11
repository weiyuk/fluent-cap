"""
Follow show_and_tell_model
https://github.com/tensorflow/models/blob/master/im2txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

import tensorflow as tf
from tensorflow.python.ops import math_ops

class LSTMModel(object):

  def __init__(self, mode, config, num_steps, model_dir,
               flag_with_saver=False,
               #model_root='./cache/models/mscoco',
               train_image_embedding=True,
               #flag_reset_state=False,
               gpu=1):
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    if self.mode != 'inference':
      self.use_weighted_loss = config.use_weighted_loss

    self.batch_size = config.batch_size
    self.num_steps = num_steps
    self.flag_with_saver = flag_with_saver
    self.train_image_embedding = train_image_embedding 
    # A float32 Tensor with shape [batch_size, vf_size].
    self._visual_features = None
    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None
    # An int32 Tensor with shape [batch_size, padded_length].
    self._input_seqs = None
    # An int32 Tensor with shape [batch_size, padded_length].
    self._target_seqs = None
    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self._input_mask = None
    # An float Tensor with shape [batch_size, padded_length] indicating sentence fluency
    self._score_weight = None
    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None
    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None
    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None
    # A float32 Tensor with shape [batch_size * padded_length].
    self._target_cross_entropy_losses = None
    # A float32 Tensor with shape [batch_size * padded_length].
    self._target_cross_entropy_loss_weights = None
    
    if self.mode == "train":
      # Set up paths and dirs
      self.model_dir = model_dir #os.path.join(model_root, model_name)
      self.variable_dir = os.path.join(self.model_dir, 'variables')
      if not os.path.exists(self.model_dir):
          os.makedirs(self.model_dir)
      if not os.path.exists(self.variable_dir):
          os.makedirs(self.variable_dir)
          
  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"
    

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.
    Outputs:
      self._visual_features
      self._input_seqs
      self._target_seqs (training and eval only)
      self._input_mask (training and eval only)
    """

    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      visual_feature = tf.placeholder(tf.float32, [self.config.vf_size], name="visual_feature")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      # images = tf.expand_dims(self.process_image(image_feed), 0)
      self._visual_features = tf.expand_dims(visual_feature, 0)
      self._input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      self._target_seqs = None
      self._input_mask = None
      self._score_weight = None
    else:
      # Inputs to the model
      self._input_seqs = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
      self._target_seqs = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
      self._visual_features = tf.placeholder(tf.float32, [self.batch_size, self.config.vf_size])
      self._input_mask = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
      self._score_weight = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
      

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.
    Inputs:
      self._visual_features
    Outputs:
      self.image_embeddings
    """
    # Map visual features into embedding space.
    if self.train_image_embedding:
      with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            inputs=self._visual_features,
            num_outputs=self.config.embedding_size,
            activation_fn=None,
            scope=scope)
    else:
      with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            trainable=False,
            inputs=self._visual_features,
            num_outputs=self.config.embedding_size,
            activation_fn=None,
            scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.
    Inputs:
      self._input_seqs
    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          )
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self._input_seqs)

    self.seq_embeddings = seq_embeddings


  def build_model(self):
    """Builds the model.
    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self._target_seqs (training and eval only)
      self._input_mask (training and eval only)
      self._score_weight (training and eval only)
    Outputs:
      self.total_loss (training and eval only)
      self._target_cross_entropy_losses (training and eval only)
      self._target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    if tf.__version__ < '1.0':
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
          num_units=self.config.num_lstm_units, state_is_tuple=True)
      if self.mode == "train":
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell,
            input_keep_prob=self.config.lstm_dropout_keep_prob,
            output_keep_prob=self.config.lstm_dropout_keep_prob)
    else:
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
      if self.mode == "train":
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=self.config.lstm_dropout_keep_prob,
            output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm") as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        if tf.__version__ < '1.0':
          tf.concat(1, initial_state, name="initial_state")
        else:
          tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        if tf.__version__ < '1.0':
          state_tuple = tf.split(1, 2, state_feed)
        else:
          state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings, squeeze_dims=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        if tf.__version__ < '1.0':   
          tf.concat(1, state_tuple, name="state")
        else:
          tf.concat(axis=1, values=state_tuple, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self._input_mask, 1)
        lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          # weights_initializer=self.initializer,
          scope=logits_scope)

    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    else:
      targets = tf.reshape(self._target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self._input_mask, [-1]))
      score_weights = tf.to_float(tf.reshape(self._score_weight, [-1]))
      #score_weights =tf.Print(score_weights,[score_weights],message='score_weights:')
      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
      if tf.__version__ < '1.0':
        if self.use_weighted_loss:
          batch_loss = tf.div(tf.reduce_sum(tf.mul(losses, score_weights)),
                          tf.reduce_sum(weights), # indicating sequence length
                          #weight_sum,
                          name="batch_loss")
        else:
          batch_loss = tf.div(tf.reduce_sum(tf.mul(losses, weights)),
                          tf.reduce_sum(weights), 
                          name="batch_loss")
      else:
        if self.use_weighted_loss:
          batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, score_weights)),
                          tf.reduce_sum(weights), # indicating sequence length
                          #weight_sum,
                          name="batch_loss")
        else:
          batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")

      if tf.__version__ < '1.0':
        tf.contrib.losses.add_loss(batch_loss)
        total_loss = tf.contrib.losses.get_total_loss()
        tf.scalar_summary("batch_loss", batch_loss)
        tf.scalar_summary("total_loss", total_loss)
        for var in tf.trainable_variables():
          tf.histogram_summary(var.op.name, var) 
      else:
        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar("batch_loss", batch_loss)
        tf.summary.scalar("total_loss", total_loss)
        for var in tf.trainable_variables():
          tf.summary.histogram(var.op.name, var)

      self.total_loss = total_loss
      self._target_cross_entropy_losses = losses  # Used in evaluation.
      self._target_cross_entropy_loss_weights = weights  # Used in evaluation.
      self._cost = cost = total_loss

  def build_saver(self): 
    # Create saver if necessary
    if self.flag_with_saver:
      self.saver = tf.train.Saver(max_to_keep=None)
      #image_embedding_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="image_embedding")
      with tf.variable_scope("image_embedding"):
        image_embedding_variables = [v for v in tf.contrib.framework.get_variables() if 'image_embedding' in v.name]
        # print ([v.name for v in image_embedding_variables])
      self.imemb_saver = tf.train.Saver(image_embedding_variables,max_to_keep=None)#{"image_embedding":image_embedding}, max_to_keep=None)

      #language model saver
      lm_variables = [v for v in tf.contrib.framework.get_variables() if 'image_embedding' not in v.name]
      self.lm_saver = tf.train.Saver(lm_variables, max_to_keep=None)

    else:
      self.saver = None

  def build_op(self):
    # Create learning rate and gradients optimizer
    config = self.config
    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    if hasattr(config, 'optimizer'):
      if config.optimizer == 'ori':
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
      elif config.optimizer == 'ada': # No GPU
        optimizer = tf.train.AdagradOptimizer(self.lr)
      elif config.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(self.lr)
      elif config.optimizer == 'rms':
        optimizer = tf.train.RMSPropOptimizer(self.lr)
      else:
        raise NameError("Unknown optimizer type %s!" % config.optimizer)
    else:
      optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.build_saver()
    if self.mode == 'train':
      self.build_op()
    
  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_seqs(self):
    return self._input_seqs

  @property
  def target_seqs(self):
    return self._target_seqs
    
  @property
  def input_mask(self):
    return self._input_mask

  @property
  def score_weight(self):
    return self._score_weight

  @property
  def visual_features(self):
    return self._visual_features

  @property
  def cost(self):
    return self._cost
  
  @property
  def final_state(self):
    return self._final_state
    
  @property
  def initial_state(self):
    return self._initial_state
  
  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
  '''  
  @property
  def embedding(self):
    return self._embedding
  '''  
  @property
  def logit(self):
    return self._logit
  
