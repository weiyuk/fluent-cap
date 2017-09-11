"""Validation based on performance for LSTM decoder"""

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
import shutil
import codecs

from caption_generator import CaptionGenerator
from inference_wrapper import InferenceWrapper
from bigfile import BigFile
from constant import *
from text import TextBank
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
flags.DEFINE_integer("word_cnt_thr", DEFAULT_WORD_COUNT, "word count threshold (default: %d)"%DEFAULT_WORD_COUNT)

flags.DEFINE_integer("ses_threads", 4, "Tensorflow CPU session threads to use")
flags.DEFINE_float("gpu_memory_fraction", 0.4, "Fraction of GPU memory to use")
flags.DEFINE_integer("gpu", 1, "select a GPU, 0:gpu0 1:gpu1 (default: 1)")

flags.DEFINE_string("model_name", DEFAULT_MODEL_NAME, "model configuration (default: %s)" % DEFAULT_MODEL_NAME)
flags.DEFINE_string("train_collection", DEFAULT_TRAIN_COLLECTION, "collection dataset for model training (default: %s)"%DEFAULT_TRAIN_COLLECTION)
flags.DEFINE_string("val_collection", DEFAULT_VAL_COLLECTION, "collection dataset for model validation (default: %s)"%DEFAULT_VAL_COLLECTION)
flags.DEFINE_string("test_collection", DEFAULT_TEST_COLLECTION, "collection dataset for model testing (default: %s)"%DEFAULT_TEST_COLLECTION)

flags.DEFINE_string("fluency_method", DEFAULT_FLUENCY_METHOD, "different ways utilizing sent_score: filter, sample, weighted, or None (default: %s)"%DEFAULT_FLUENCY_METHOD)

flags.DEFINE_string("checkpoint_style", "file", "file, iter_interval, iter_num")
flags.DEFINE_string("eval_start", "196000-200000-4000", "start_iter end_iter step_iter")
flags.DEFINE_integer("top_k", 10, "number of models to be evaluated")
flags.DEFINE_integer("iter_num", 2000, "iteration number")

flags.DEFINE_string("vf_name", DEFAULT_VISUAL_FEAT, "name of the visual feature (default: %s)"%DEFAULT_VISUAL_FEAT)
flags.DEFINE_integer("beam_size", DEFAULT_BEAM_SIZE, "beam search size (default: %d)" % DEFAULT_BEAM_SIZE)
flags.DEFINE_float("length_normalization_factor", 0, "length normalization factor to encourage longer sentence")
flags.DEFINE_integer("overwrite", 0, "overwrite existing file (default: 0)")
#flags.DEFINE_integer("video", 0, "is video data (default: 0)")

FLAGS = flags.FLAGS
rootpath = FLAGS.rootpath


def main(unused_args):

  length_normalization_factor = FLAGS.length_normalization_factor

  # Load model configuration
  config_path = os.path.join(os.path.dirname(__file__), 'model_conf', FLAGS.model_name + '.py')
  config = utility.load_config(config_path)

  config.trainCollection = FLAGS.train_collection
  config.word_cnt_thr = FLAGS.word_cnt_thr
  config.rootpath = FLAGS.rootpath

  train_collection =  FLAGS.train_collection
  test_collection = FLAGS.test_collection
  overwrite = FLAGS.overwrite
  feature = FLAGS.vf_name


  img_set_file = os.path.join(rootpath, test_collection, 'VideoSets', '%s.txt' % test_collection)
  if not os.path.exists(img_set_file):
      img_set_file = os.path.join(rootpath, test_collection, 'ImageSets', '%s.txt' % test_collection)
  img_list = map(str.strip, open(img_set_file).readlines())

  # have visual feature ready
  FLAGS.vf_dir = os.path.join(rootpath, test_collection, 'FeatureData', feature)
  vf_reader = BigFile( FLAGS.vf_dir )

  textbank = TextBank(utility.get_train_vocab_file(FLAGS))
  config.vocab_size = len(textbank.vocab)
  config.vf_size = int(open(os.path.join(FLAGS.vf_dir, 'shape.txt')).read().split()[1])

  model_dir = utility.get_model_dir(FLAGS)
  output_dir = utility.get_pred_dir(FLAGS)

  checkpoint_style = FLAGS.checkpoint_style

  if checkpoint_style == 'file':
    #output_per_filename = 'model_perf_in_topk_%d_%s' % (FLAGS.top_k, FLAGS.eval_model_list_file)
    # read validated top models
    validation_output_dir = utility.get_sim_dir(FLAGS)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_model_list_file = os.path.join(validation_output_dir, 'loss_info.txt') #FLAGS.eval_model_list_file)
    shutil.copy(eval_model_list_file, output_dir)
    test_iter_list = []
    for line in open(eval_model_list_file).readlines()[:FLAGS.top_k]:
      iter_current = int(line.strip().split()[0])
      test_iter_list.append(iter_current)

  elif checkpoint_style == 'iter_interval':
    #output_per_filename =  'model_perf_in_%s' % FLAGS.eval_stat
    test_iter_list = range(*[int(x) for x in FLAGS.eval_stat.split("-")])
  elif checkpoint_style == 'iter_num':
    #output_per_filename =  'model_perf_in_iter_%d' % FLAGS.iter_num
    test_iter_list = [FLAGS.iter_num]

  with_image_embedding = True if FLAGS.with_image_embedding != 0 else False
  g = tf.Graph()
  with g.as_default():
    model = InferenceWrapper(config=config,model_dir=model_dir,
                             gpu_memory_fraction=FLAGS.gpu_memory_fraction,
                             gpu=FLAGS.gpu,
                             with_image_embedding=with_image_embedding)
    model.build_model()
  
  for k, iter_n in enumerate(test_iter_list):
    model_path = os.path.join(model_dir, 'variables', 'model_%d.ckpt' % iter_n)
    while not os.path.exists(model_path+'.meta'):
      logger.error('Model path: %s', model_path)
      logger.error('Cannot load model file and exit')
      sys.exit(0)

    top_one_pred_sent_file = os.path.join(output_dir, 'top%d' % k, 'top_one_pred_sent.txt')
    top_n_pred_sent_file = os.path.join(output_dir, 'top%d' % k, 'top_n_pred_sent.txt')
    # perf_file = os.path.join(output_dir, 'model_%d.ckpt' % iter_n, 'perf.txt')

    if os.path.exists(top_one_pred_sent_file) and not overwrite:
      # write existing perf file and print out
      logger.info('%s exists. skip', top_one_pred_sent_file)
      continue

    if not os.path.exists(os.path.split(top_one_pred_sent_file)[0]):
      os.makedirs(os.path.split(top_one_pred_sent_file)[0])

    logger.info('save results to %s', top_one_pred_sent_file)

    # load the trained model
    generator = CaptionGenerator(config, model, length_normalization_factor = length_normalization_factor)
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config_proto = tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.ses_threads, gpu_options=gpu_options, allow_soft_placement=True)
    #with  tf.Session(config=config_proto) as session:
      #model.build_model(session, model_path)
    model.load_model(model_path)

    fout_one_sent = codecs.open(top_one_pred_sent_file, 'w','utf-8')
    fout_n_sent = codecs.open(top_n_pred_sent_file, 'w','utf-8')

    for progress,img in enumerate(img_list):
        # predict sentences given a visual feature
        visual_feature = np.array(vf_reader.read_one(img))
        sentences = generator.beam_search( visual_feature, FLAGS.beam_size)

        # output top one sentence info
        sent_score = sentences[0].score
        sent = ' '.join(sentences[0].words)
        fout_one_sent.write(img + ' ' + '%.3f' % sent_score + ' ' + sent + '\n')
        logger.debug(img + ' ' + '%.3f' % sent_score + ' ' + sent)

        # output top n sentences info
        fout_n_sent.write(img)
        for sentence in sentences:
            sent_score = sentence.score
            sent = ' '.join(sentence.words)
            fout_n_sent.write('\t' + '%.3f' % sent_score + '\t' + sent)
        fout_n_sent.write('\n')
      
        if progress % 100 == 0:
          logger.info('%d images decoded' % (progress+1))

    logger.info('%d images decoded' % (progress+1))
 
    fout_one_sent.close()
    fout_n_sent.close()


if __name__ == "__main__":
  tf.app.run()
