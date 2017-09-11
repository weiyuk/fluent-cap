"""Decoder (sentence generator) for the trained mRNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import logging
import copy
import heapq
import math

import tensorflow as tf

from constant import TOKEN_BOS
from text import TextBank
from utility import get_vocab_file

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class Caption(object):
  """Represents a complete or partial caption."""

  def __init__(self, sentence, state, logprob, score, metadata=None, words=None):
    """Initializes the Caption.
    Args:
      sentence: List of word ids in the caption.
      state: Model state after generating the previous word.
      logprob: Log-probability of the caption.
      score: Score of the caption.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'sentence'.
    """
    self.sentence = sentence # indexes list
    self.words = words
    self.state = state
    self.logprob = logprob
    self.score = score
    self.metadata = metadata

  def __cmp__(self, other):
    """Compares Captions by score."""
    assert isinstance(other, Caption)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1

class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.
    The only method that can be called immediately after extract() is reset().
    Args:
      sort: Whether to return the elements in descending sorted order.
    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []



class CaptionGenerator(object):
  """Class to generate captions from an image-to-text model."""

  def __init__(self, config, model, length_normalization_factor=0.0):
    self.config = copy.deepcopy(config)
    self.config.batch_size = 1
    self.model = model

    self.textbank = TextBank(get_vocab_file(config.trainCollection, config.word_cnt_thr, config.rootpath))
    self.length_normalization_factor=length_normalization_factor
    

  def beam_search(self, visual_feature, beam_size, max_steps=30, tag2score=None):
    """Decode an image with a sentences."""
    assert visual_feature.shape[0] == self.config.vf_size
    #assert self.flag_load_model, 'Must call local_model first'
    # Get the initial logit and state
    initial_state = self.model.feed_visual_feature(visual_feature)

    initial_beam = Caption(
        sentence=[self.textbank.vocab[TOKEN_BOS]],
        state=initial_state[0],
        logprob=0.0,
        score=0.0,
        metadata=[""])
    partial_captions = TopN(beam_size)
    partial_captions.push(initial_beam)
    complete_captions = TopN(beam_size)

    # Run beam search.
    for _ in range(max_steps - 1): 
      partial_captions_list = partial_captions.extract()
      partial_captions.reset()
      input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
      state_feed = np.array([c.state for c in partial_captions_list])
      softmax, new_states, metadata = self.model.inference_step(#sess,
                                                                input_feed,
                                                                state_feed)
      for i, partial_caption in enumerate(partial_captions_list):
        word_probabilities = softmax[i]
        state = new_states[i]
        # For this partial caption, get the beam_size most probable next words.
        words_and_probs = list(enumerate(word_probabilities))
        words_and_probs.sort(key=lambda x: -x[1])
        words_and_probs = words_and_probs[0:beam_size]
        # Each next word gives a new partial caption.
        for w, p in words_and_probs:
          if tag2score!=None and w in tag2score and w not in partial_caption.sentence:
            p+=tag2score[w]
          if p < 1e-12:
            continue  # Avoid log(0).
          sentence = partial_caption.sentence + [w]
          logprob = partial_caption.logprob + math.log(p)
          score = logprob
          if metadata:
            metadata_list = partial_caption.metadata + [metadata[i]]
          else:
            metadata_list = None
          if w == self.textbank.vocab[TOKEN_BOS]:
            if self.length_normalization_factor > 1e-6:
              score /= len(sentence)**self.length_normalization_factor
            beam = Caption(sentence, state, logprob, score, metadata_list)
            complete_captions.push(beam)
          else:
            beam = Caption(sentence, state, logprob, score, metadata_list)
            partial_captions.push(beam)
      if partial_captions.size() == 0:
        # We have run out of partial candidates; happens when beam_size = 1.
        break

    # If we have no complete captions then fall back to the partial captions.
    # But never output a mixture of complete and partial captions because a
    # partial caption could have a higher score than all the complete captions.
    if not complete_captions.size():
      complete_captions = partial_captions

    captions = complete_captions.extract(sort=True)
    for i, caption in enumerate(captions):
        caption.words = self.textbank.decode_tokens(caption.sentence[1:])

    return captions


if __name__ == '__main__':
    from utility import load_config
    from constant import ROOT_PATH, DEFAULT_WORD_COUNT
    config = load_config('model_conf/8k_neuraltalk.py')
    config.trainCollection = 'flickr8kenctrain'
    config.word_cnt_thr = DEFAULT_WORD_COUNT
    config.rootpath = ROOT_PATH
    model = None
    generator = CaptionGenerator(config, model)

