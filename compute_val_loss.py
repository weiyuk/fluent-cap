"""Validation based on loss for LSTM decoder"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import logging
import json
import tensorflow as tf

from sampled_data_provider import BucketDataProvider
from lstm_model import LSTMModel
from text import TextBank
from constant import *
from bigfile import BigFile
import utility

logger = logging.getLogger(__file__)
formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format=formatter_log,
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

flags = tf.flags
flags.DEFINE_string("rootpath", ROOT_PATH, "rootdir of the data and model (default: %s)"%ROOT_PATH)
flags.DEFINE_integer("with_image_embedding", 1, "With image embedding layer? (default:1)")
flags.DEFINE_integer("ses_threads", 4, "Tensorflow CPU session threads to use")
flags.DEFINE_float("gpu_memory_fraction", 0.4, "Fraction of GPU memory to use")
flags.DEFINE_integer("gpu", 1, "select a GPU, 0:gpu0 1:gpu1 (default: 1)")

flags.DEFINE_string("model_name", DEFAULT_MODEL_NAME, "model configuration (default: %s)" % DEFAULT_MODEL_NAME)
flags.DEFINE_string("train_collection", DEFAULT_TRAIN_COLLECTION, "collection dataset for model training (default: %s)"%DEFAULT_TRAIN_COLLECTION)
flags.DEFINE_string("val_collection", DEFAULT_VAL_COLLECTION, "collection dataset for model validation (default: %s)"%DEFAULT_VAL_COLLECTION)
flags.DEFINE_integer("word_cnt_thr", DEFAULT_WORD_COUNT, "word count threshold (default: %d)"%DEFAULT_WORD_COUNT)

flags.DEFINE_string("fluency_method", DEFAULT_FLUENCY_METHOD, "different ways utilizing sent_score: filter, sample, weighted, or None (default: %s)"%DEFAULT_FLUENCY_METHOD)
flags.DEFINE_string("vf_name", DEFAULT_VISUAL_FEAT, "name of the visual feature (default: %s)"%DEFAULT_VISUAL_FEAT)
flags.DEFINE_integer("language", DEFAULT_LANG, "language, 0: English, 1: Chinese (default: %d)" % DEFAULT_LANG)

flags.DEFINE_integer("overwrite", 0, "overwrite existing file (default: 0)")

FLAGS = flags.FLAGS
rootpath = FLAGS.rootpath


def run_epoch(session, batch_size, bucket_size, config, model, data_provider):
  """Runs the model on the given data."""
  costs = 0.0
  
  for iters, (ind_buc, x, y, vf, fg, sent_score) in enumerate(
    data_provider.generate_batches(batch_size, [bucket_size])):

    # mutiply the input mask(1/0) with sent_scores
    score_weight = np.array(fg )
    if config.use_weighted_loss:
      temp = np.zeros(fg.shape, dtype=np.float32)
      for i in range(fg.shape[0]):
        temp[i,:] = fg[i,:] * sent_score[i]
      score_weight = temp

    # run forward propgation
    cost = session.run(model.cost,
                          {model.input_seqs: x,
                           model.target_seqs: y,
                           model.visual_features: vf,
                           model.score_weight: score_weight,
                           model.input_mask: fg})
                           
    costs += cost

  return (costs / iters)



def main(unused_args):
  train_collection =  FLAGS.train_collection
  val_collection = FLAGS.val_collection
  overwrite = FLAGS.overwrite
  output_dir = utility.get_sim_dir(FLAGS)
  loss_info_file = os.path.join(output_dir, 'loss_info.txt')
  if os.path.exists(loss_info_file) and not overwrite:
      logger.info('%s exists. quit', loss_info_file)
      sys.exit(0)

  model_dir=utility.get_model_dir(FLAGS)
  config_path = os.path.join(os.path.dirname(__file__), 'model_conf', FLAGS.model_name + '.py')
  config = utility.load_config(config_path)

  config.fluency_method = FLAGS.fluency_method
  if config.fluency_method == 'weighted':
    config.use_weighted_loss = True
  else:
    config.use_weighted_loss = False

  textbank = TextBank(utility.get_train_vocab_file(FLAGS))
  config.vocab_size = len(textbank.vocab)
  config.vf_size = int(open(os.path.join(utility.get_val_feat_dir(FLAGS), 'shape.txt')).read().split()[1])

  gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
  config_proto = tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.ses_threads, gpu_options=gpu_options, allow_soft_placement=True)

  with_image_embedding = True if FLAGS.with_image_embedding > 0 else False
  with tf.Graph().as_default(), tf.Session(config=config_proto) as session:

    assert len(config.buckets) >= 1
    assert config.buckets[-1] == config.max_num_steps
    with tf.device('/gpu:%d'%FLAGS.gpu):
      with tf.variable_scope("LSTMModel", reuse=None):
        if with_image_embedding:
          model = LSTMModel(mode='eval',
                num_steps=config.buckets[-1], 
                config=config,
                model_dir=model_dir, #model_name=FLAGS.model_name,
                flag_with_saver=True)
                #model_root=FLAGS.model_root)
        else:
          # deprecating this option
          print('Plz use image_embedding')
          sys.exit(-1)          
        model.build()    


    model_path_list = []
    _dir = os.path.join(model_dir,'variables')
    for _file in os.listdir(_dir):
      if _file.startswith('model_') and _file.endswith('.ckpt.meta'):
        iter_n = int(_file[6:-10])
        model_path = os.path.join(_dir, 'model_%d.ckpt'%iter_n)
        model_path_list.append((iter_n, model_path))

    data_provider = BucketDataProvider(val_collection, utility.get_train_vocab_file(FLAGS), 
          feature=FLAGS.vf_name, 
          language=FLAGS.language, 
          flag_shuffle=False, 
          method=config.fluency_method,
          rootpath=rootpath)
    iter2loss = {}
    for iter_n, model_path in model_path_list:
      loss_file = os.path.join(output_dir, 'model_%d.ckpt' % iter_n, 'loss.txt')
      if os.path.exists(loss_file) and not overwrite:
          logger.info('load loss from %s', loss_file)
          loss = float(open(loss_file).readline().strip())
          iter2loss[iter_n] = loss
          continue
      if not os.path.exists(os.path.split(loss_file)[0]):
          os.makedirs(os.path.split(loss_file)[0])

      model.saver.restore(session, model_path)
      # print([v.name for v in tf.trainable_variables()])
      logger.info('Continue to train from %s', model_path)

      val_cost = run_epoch(session, config.batch_size, config.buckets[-1], config,model, data_provider)
      logger.info("Validation cost for checkpoint model_%d.ckpt is %.3f" % (iter_n, val_cost))

      iter2loss[iter_n] = val_cost
      with open(loss_file, "w") as fw:
        fw.write('%g' % val_cost)
        fw.close()

  sorted_iter2loss = sorted(iter2loss.iteritems(), key=lambda x: x[1])
  with open(loss_info_file, 'w') as fw:
      fw.write('\n'.join(['%d %g' % (iter_n, loss) for (iter_n, loss) in sorted_iter2loss]))
      fw.close()


if __name__ == "__main__":
  tf.app.run()
