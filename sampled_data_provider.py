from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import re
import string
import time
import random
import codecs

from constant import ROOT_PATH, DEFAULT_LANG, DEFAULT_FLUENCY_U, TOKEN_PAD, TOKEN_BOS
from bigfile import BigFile
from text import TextTool, TextBank
import utility

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)




class Batch(object):
    def __init__(self, batch_size, max_seq_len, vf_size, bos_ind, 
                 fluency_threshold=DEFAULT_FLUENCY_U):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vf_size = vf_size
        self.bos_ind = bos_ind
        self.fluency_threshold = fluency_threshold
        self.empty()
      
    def empty(self):
        self.x = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
        self.y = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
        self.vf = np.zeros([self.batch_size, self.vf_size], dtype=np.float32)
        self.fg = np.zeros([self.batch_size, self.max_seq_len], dtype=np.float32)
        self.sent_score = np.zeros([self.batch_size], dtype=np.float32)
        self.num_feed = 0
        
    def feed_and_vomit(self, visual_feature, sentence, score=None):
      i = self.num_feed
      # feed sentence
      self.x[i, 0] = self.bos_ind
      if len(sentence) > self.max_seq_len - 1:
          self.x[i, 1:] = sentence[:self.max_seq_len-1]
          self.y[i, :self.max_seq_len-1] = sentence[:self.max_seq_len-1]
          self.y[i, self.max_seq_len-1] = self.bos_ind
          self.fg[i, :] = np.ones([self.max_seq_len], dtype=np.float32)
      else:
          l = len(sentence)
          self.x[i, 1:l+1] = sentence
          self.y[i, :l] = sentence
          self.y[i, l] = self.bos_ind
          self.fg[i, :l+1] = np.ones([l+1], dtype=np.float32)
      
      if score != None:
         self.sent_score[i] = 1.0 if score >= self.fluency_threshold else score
      # feed visual feature
      assert visual_feature.shape[0] == self.vf_size
      self.vf[i, :] = visual_feature
      self.num_feed += 1
      assert self.num_feed <= self.batch_size
      # vomit if necessary
      if self.num_feed == self.batch_size:
          return (self.x, self.y, self.vf, self.fg, self.sent_score)
      return None


class BucketDataProvider(object):
    """TensorFlow Data Provider with Buckets"""
    def __init__(self, collection, vocab_file, feature, language,
                flag_shuffle=True, method=None, fluency_threshold=DEFAULT_FLUENCY_U, rootpath=ROOT_PATH):
        self.language = language
        self.anno_file_path = utility.get_sent_file(collection, language, rootpath)
        self.sent_score_file = utility.get_sent_score_file(collection, language, rootpath)
        self.fluency_threshold = fluency_threshold
        self.method = method
        if method:
            assert method in ['sample','filter','weighted']
            assert self.sent_score_file != None
            assert fluency_threshold>0
            if method == 'weighted':
                # Not sampling the data if fluency-guided method is weighted_loss
                self.method = method = None 

        self.textbank = TextBank(vocab_file)
        assert self.textbank.vocab[TOKEN_PAD] == 0
        self.vf_reader = BigFile(utility.get_feat_dir(collection, feature, rootpath))
        self.vf_names = set(self.vf_reader.names)
        self.vf_size = self.vf_reader.ndims
        self.flag_shuffle = flag_shuffle
        self._load_data()


    def shuffle_data_queue(self):
        random.shuffle(self._data_queue)


    def generate_batches(self, batch_size, buckets):
        """Return a list generator of mini-batches of training data."""
        # create Batches
        batches = []
        for max_seq_len in buckets:
            batches.append(Batch(batch_size, max_seq_len, self.vf_size, self.textbank.vocab[TOKEN_BOS]))
        
        # shuffle if necessary
        if self.flag_shuffle:
            np.random.shuffle(self._data_queue)
        # scan data queue
        for data in self._data_queue:
            if self.method:
                if data['sent_score'] < self.fluency_threshold:
                    if self.method == 'filter':
                        #Drop if the sent_score < threshold
                        continue
                    elif self.method == 'sample':
                        # Drop with certain probability if the sent_score < 1
                        x = random.uniform(0, self.fluency_threshold)
                        if x > data['sent_score']:
                            continue
            score = data['sent_score'] if self.sent_score_file else None
            sentence = data['sentence']
            # Load visual features
            visual_features = np.array(self.vf_reader.read_one(data['image_id']))
            if len(sentence) >= buckets[-1]:
                feed_res = batches[-1].feed_and_vomit(visual_features, sentence, score)
                ind_buc = len(buckets) - 1
            else:
                for (ind_b, batch) in enumerate(batches):
                    if len(sentence) < batch.max_seq_len:
                        feed_res = batches[ind_b].feed_and_vomit(visual_features, sentence, score)
                        ind_buc = ind_b
                        break
            if feed_res:
                yield (ind_buc,) + feed_res
                batches[ind_buc].empty()
            
    def _load_data(self, verbose=True):
        logger.debug('Loading data')
        self._data_queue = []
        ind_img = 0
        num_failed = 0
        if self.sent_score_file != None:
            sid2score={}
            for line in open(self.sent_score_file):
                elem = line.strip().split('\t')
                sid = elem[0]
                score= float(elem[-1])
                sid2score[sid] = score
        annos = codecs.open(self.anno_file_path,'r','utf-8').readlines()
        for (ind_a, line) in enumerate(annos):
            data = {}
            sid, sent = line.strip().split(" ", 1)
            imgid = sid.strip().split("#")[0]
            if imgid.endswith('.jpg')  or imgid.endswith('.mp4'):
                imgid = imgid[:-4]
            #assert imgid in self.vf_names, '%s not in feature data'%imgid
            assert(imgid in self.vf_names)
            #if imgid not in self.vf_names:
            #    logger.info('%s not in feature data, skipping that.'%imgid)
            #    continue
            data['image_id'] = imgid
            
            # Encode sentences
            tokens = TextTool.tokenize(sent, self.language)
            data['sentence'] = self.textbank.encode_tokens(tokens, flag_add_bos=False)
            data['sent_score'] = sid2score[sid] if self.sent_score_file and sid in sid2score else 1
            self._data_queue.append(data)
            if verbose and (ind_a + 1) % 20000 == 0:
                logger.debug('%d/%d annotation', ind_a + 1, len(annos))
        random.shuffle( self._data_queue )
        
        nr_of_images = len(set([data['image_id'] for data in self._data_queue]))
        logger.info('%d images, %d sentences from %s', nr_of_images, len(self._data_queue), self.anno_file_path)

if __name__ == '__main__':
    from utility import get_vocab_file
    rootpath = ROOT_PATH
    collection = 'flickr8kenctrain'
    collection = 'flickr8kzhbJanbosontrain'
    #collection = 'flickr8kzh'
    word_cnt_thr = 5
    feature = 'pygooglenet_bu4k-pool5_7x7_s1'
    data_provider = BucketDataProvider(collection, get_vocab_file(collection), feature, language=1, rootpath=rootpath)
    batch_size = 100
    buckets = [16]

    for step, (ind_buc, x, y, vf, fg, sent_score) in enumerate(data_provider.generate_batches(batch_size, buckets)):
        print (step, ind_buc, x.shape, vf.shape, sent_score.shape)
        break
  

