"""Trainer for visual captioning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.ops import math_ops

from constant import *
from text import TextTool, TextBank
from sampled_data_provider import BucketDataProvider
from lstm_model import LSTMModel
import utility

logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


flags = tf.flags
flags.DEFINE_string("rootpath", ROOT_PATH, "rootdir of the data and model (default: %s)"%ROOT_PATH)
flags.DEFINE_integer("with_image_embedding", 1, "With image embedding layer? (default:1)")
flags.DEFINE_integer("ses_threads", 2, "Tensorflow CPU session threads to use")
flags.DEFINE_float("gpu_memory_fraction", 0.3, "Fraction of GPU memory to use")
flags.DEFINE_integer("gpu", 1, "select a GPU, 0:gpu0 1:gpu1 (default: 1)")

flags.DEFINE_string("model_name", DEFAULT_MODEL_NAME, "model configuration (default: %s)" % DEFAULT_MODEL_NAME)

flags.DEFINE_string("train_collection", DEFAULT_TRAIN_COLLECTION, "collection dataset for model training (default: %s)"%DEFAULT_TRAIN_COLLECTION)
flags.DEFINE_integer("word_cnt_thr", DEFAULT_WORD_COUNT, "word count threshold (default: %d)"%DEFAULT_WORD_COUNT)

flags.DEFINE_string("fluency_method", DEFAULT_FLUENCY_METHOD, "different ways utilizing sent_score: filter, sample, weighted, or None (default: %s)"%DEFAULT_FLUENCY_METHOD)

flags.DEFINE_string("vf_name", DEFAULT_VISUAL_FEAT, "name of the visual feature (default: %s)"%DEFAULT_VISUAL_FEAT)
flags.DEFINE_integer("language", DEFAULT_LANG, "language, 0: English, 1: Chinese (default: %d)" % DEFAULT_LANG)
flags.DEFINE_string("pre_trained_model_path", "", "path of the pre_trained model, if empty will train from scratch.")

flags.DEFINE_string("pre_trained_imembedding_path", "", "path of the pre_trained image_embedding, if empty will initialize randomly.")
flags.DEFINE_string("pre_trained_lm_path", "", "path of the pre_trained language model, if empty will initialize randomly.")

flags.DEFINE_integer("overwrite", 0, "overwrite existing file (default: 0)")

FLAGS = flags.FLAGS


def run_epoch(session, iters_done, config, models, data_provider, 
    verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0

  # Determine the learning rate with lr decay
  lr_decay_dstep = max(0, 
      (iters_done - config.lr_decay_keep) // config.lr_decay_iter)
  lr_decay = config.lr_decay ** lr_decay_dstep
  for m in models:
    m.assign_lr(session, config.learning_rate * lr_decay)
    
  for step, (ind_buc, x, y, vf, fg, sent_score) in enumerate(
      data_provider.generate_batches(config.batch_size, config.buckets)):
    # update the lr if necessary
    lr_decay_dstep_cur = max(0, 
        (iters_done + step - config.lr_decay_keep) // config.lr_decay_iter)
    if lr_decay_dstep_cur > lr_decay_dstep:
      lr_decay_dstep = lr_decay_dstep_cur
      lr_decay = config.lr_decay ** lr_decay_dstep
      for m in models:
        m.assign_lr(session, config.learning_rate * lr_decay)

    score_weight = np.array(fg )
    if config.use_weighted_loss:
      temp = np.zeros(fg.shape, dtype=np.float32)
      for i in range(fg.shape[0]):
        temp[i,:] = fg[i,:] * sent_score[i]
      score_weight = temp

    # run forward and backward propgation
    m = models[ind_buc]
    cost, _ = session.run([m.cost, m.train_op],
                          {m.input_seqs: x,
                           m.target_seqs: y,
                           m.visual_features: vf,
                           m.score_weight: score_weight,
                           m.input_mask: fg})
                           
    costs += cost
    iters += 1

    # print loss if necessary
    if verbose and (iters_done + iters) % config.num_iter_verbose == 0:
      logger.info("Step %d, lr %.6f, model bucket %d(%d)"
                  ": Avg/Cur cost: %.3f/%.3f speed: %.0f sps" %
                  (iters + iters_done, config.learning_rate * lr_decay, 
                   ind_buc, config.buckets[ind_buc],
                   costs / iters, cost, 
                   iters * config.batch_size / (time.time() - start_time)))
      
    # save the current model if necessary
    '''if (iters_done + iters) % config.num_iter_save == 0:
      models[0].saver.save(session, os.path.join(m.variable_dir, 
          'model_%d.ckpt' % (iters_done + iters)))
      models[0].imemb_saver.save(session, os.path.join(m.variable_dir,
          'imembedding_model_%d.ckpt' % (iters_done + iters)))
      logger.info("Model saved with itereation %d", iters_done + iters)
    '''
  return (costs / iters, iters_done + iters)


def main(unused_args):
  model_dir=utility.get_model_dir(FLAGS)
  if os.path.exists(model_dir) and not FLAGS.overwrite:
    logger.info('%s exists. quit', model_dir)
    sys.exit(0)

  # Load model configuration
  config_path = os.path.join(os.path.dirname(__file__), 'model_conf', FLAGS.model_name + '.py')
  config = utility.load_config(config_path)
  
  rootpath = FLAGS.rootpath
  train_collection = FLAGS.train_collection
  feature = FLAGS.vf_name
  
  vf_dir = utility.get_feat_dir(train_collection, feature, rootpath) 
  vocab_file = utility.get_vocab_file(train_collection, FLAGS.word_cnt_thr, rootpath)
  textbank = TextBank(vocab_file)
  config.vocab_size = len(textbank.vocab)
  config.vf_size = int(open(os.path.join(vf_dir, 'shape.txt')).read().split()[1])

  if hasattr(config,'num_epoch_save'):
    num_epoch_save = config.num_epoch_save
  else:
    num_epoch_save = 1

  if FLAGS.fluency_method == 'None':
      FLAGS.fluency_method = None
  config.fluency_method = FLAGS.fluency_method
  if config.fluency_method == 'weighted':
    config.use_weighted_loss = True
  else:
    config.use_weighted_loss = False

  train_image_embedding = True
  try:
    if config.train_image_embedding == False:
      assert('freeze' in FLAGS.model_name)
      train_image_embedding = False 
      logger.info('Not training image embedding')
  except:
    pass

  with_image_embedding = True if FLAGS.with_image_embedding != 0 else False
  # Start model training
  gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config_proto = tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.ses_threads, gpu_options=gpu_options, allow_soft_placement=True)
 
  with tf.Graph().as_default(), tf.Session(config=config_proto) as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    assert len(config.buckets) >= 1
    assert config.buckets[-1] == config.max_num_steps
    models = []
    with tf.device('gpu:%s'%FLAGS.gpu):
      with tf.variable_scope("LSTMModel", reuse=None, initializer=initializer):
        if with_image_embedding:
          m = LSTMModel(mode='train',
              num_steps=config.buckets[0], 
              config=config,
              model_dir=model_dir,
              flag_with_saver=True,
              train_image_embedding=train_image_embedding)
              #model_root=FLAGS.model_root)
        else:
          # deprecating this function
          logger.info('Plz use with_image_embedding=1')
          sys.exit(-1)
          '''m = PCALSTMModel(mode='train',
              num_steps=config.buckets[0],
              config=config,
              model_name=FLAGS.model_name,
              flag_with_saver=True,
              train_image_embedding=train_image_embedding,
              model_root=FLAGS.model_root)
          '''
        m.build()
        models.append(m)

    pre_trained_iter=0 
    if FLAGS.pre_trained_model_path:
      pre_trained_iter = int(FLAGS.pre_trained_model_path.split('model_')[1].split('.')[0])
    hdlr = logging.FileHandler(os.path.join(m.model_dir, 'log%d.txt'%pre_trained_iter))
    hdlr.setLevel(logging.INFO)
    hdlr.setFormatter(logging.Formatter(formatter_log))
    logger.addHandler(hdlr)

    if FLAGS.pre_trained_model_path:
      if tf.__version__ < '1.0':
        tf.initialize_all_variables().run()
      else:
        tf.global_variables_initializer().run()
      models[0].saver.restore(session, FLAGS.pre_trained_model_path)
      logger.info('Continue to train from %s', FLAGS.pre_trained_model_path)
    elif FLAGS.pre_trained_imembedding_path:
      if tf.__version__ < '1.0':
        tf.initialize_all_variables().run()
      else:
        tf.global_variables_initializer().run()
      models[0].imemb_saver.restore(session, FLAGS.pre_trained_imembedding_path)
      logger.info('Init image-embedding from %s', FLAGS.pre_trained_imembedding_path)
    elif FLAGS.pre_trained_lm_path:
      if tf.__version__ < '1.0':
        tf.initialize_all_variables().run()
      else:
        tf.global_variables_initializer().run()
      models[0].lm_saver.restore(session, FLAGS.pre_trained_lm_path)
      logger.info('Init language from %s', FLAGS.pre_trained_lm_path)
    else:
      if tf.__version__ < '1.0':
        tf.initialize_all_variables().run()
      else:
        tf.global_variables_initializer().run()
      # print([v.name for v in tf.trainable_variables()])

    iters_done = 0
    data_provider = BucketDataProvider(FLAGS.train_collection, vocab_file, FLAGS.vf_name, 
                               language=FLAGS.language, method=config.fluency_method, 
                               rootpath=FLAGS.rootpath)
    
    for i in range(config.num_epoch):
      logger.info('epoch %d', i)
      data_provider.shuffle_data_queue()
      train_cost, iters_done = run_epoch(session, iters_done, config, models, data_provider, verbose=True)
      logger.info("Train cost for epoch %d is %.3f" % (i, train_cost))

      # save the current model if necessary
      if (i+1)% num_epoch_save == 0:
          models[0].saver.save(session, os.path.join(m.variable_dir,
                'model_%d.ckpt' % (iters_done+pre_trained_iter)))
          if with_image_embedding: 
              models[0].imemb_saver.save(session, os.path.join(m.variable_dir, \
                 'imembedding_model_%d.ckpt' % (iters_done)))
          logger.info("Model saved at iteration %d", iters_done)


  # copy the configure file in to checkpoint direction
  os.system("cp %s %s" % (config_path, model_dir))
  if FLAGS.pre_trained_model_path:
    os.system("echo %s > %s" % (FLAGS.pre_trained_model_path, os.path.join(model_dir, 'pre_trained_model_path.txt')))
  if FLAGS.pre_trained_imembedding_path:
    os.system("echo %s > %s" % (FLAGS.pre_trained_imembedding_path, os.path.join(model_dir, 'pre_trained_imembedding_path.txt')))



if __name__ == "__main__":
  tf.app.run()
