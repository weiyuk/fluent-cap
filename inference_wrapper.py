# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model wrapper class for performing inference with a ShowAndTellModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy
import logging
import tensorflow as tf

from lstm_model import LSTMModel

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class InferenceWrapper(object):
  """Model wrapper class for performing inference with a ShowAndTellModel."""

  def __init__(self, config, model_dir,
               ses_threads=2,
               length_normalization_factor=0.0,
               gpu_memory_fraction=0.3,
               gpu=1,
               with_image_embedding=True):
    self.config = copy.deepcopy(config)
    self.config.batch_size = 1
    self.flag_load_model = False
    self.model_dir = model_dir
    self.gpu= gpu
    self.gpu_memory_fraction = gpu_memory_fraction
    self.with_image_embedding = with_image_embedding

  def build_model(self):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction)
    config_proto = tf.ConfigProto( gpu_options=gpu_options, allow_soft_placement=True)
    self.session = session = tf.Session(config=config_proto) 
    with tf.device('/gpu:%d'%self.gpu):
      with tf.variable_scope("LSTMModel", reuse=None):
        if self.with_image_embedding:
          self.model = LSTMModel(config=self.config, mode="inference", 
                      model_dir = self.model_dir,
                      flag_with_saver=True,
                      num_steps = 1,
                      gpu=self.gpu)
        else:
          print ('Please use with_image_embeddind=1')
          sys.exit(-1)
        self.model.build()

  def load_model(self, model_path):
      self.model.saver.restore(self.session, model_path)
      self.flag_load_model = True
      self.model_path = model_path
      logger.info('Load model from %s', model_path)

  def feed_visual_feature(self, visual_feature):
    assert visual_feature.shape[0] == self.config.vf_size
    #assert self.flag_load_model, 'Must call local_model first'
    sess = self.session
    initial_state = sess.run(fetches="LSTMModel/lstm/initial_state:0",
                             feed_dict={"LSTMModel/visual_feature:0": visual_feature})
    return initial_state

  def inference_step(self, input_feed, state_feed):
    sess = self.session
    softmax_output, state_output = sess.run(
        fetches=["LSTMModel/softmax:0", "LSTMModel/lstm/state:0"],
        feed_dict={
            "LSTMModel/input_feed:0": input_feed,
            "LSTMModel/lstm/state_feed:0": state_feed,
        })
    return softmax_output, state_output, None

if __name__ == '__main__':
    from utility import load_config
    from constant import ROOT_PATH, DEFAULT_WORD_COUNT
    config = load_config('model_conf/8k_neuraltalk.py')
    config.trainCollection = 'flickr8kenctrain'
    config.word_cnt_thr = DEFAULT_WORD_COUNT
    config.rootpath = ROOT_PATH
    model_name = None
    model = InferenceWrapper(config, model_name)

