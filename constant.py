import os

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
DEFAULT_WORD_COUNT = 5
DEFAULT_LANG = 1  # 0: English, 1: Chinese, ...
DEFAULT_FLUENCY_U = 0.5
TOKEN_PAD = '<pad>'
TOKEN_UNK = '<unk>'
TOKEN_BOS = '<bos>'
DEFAULT_TRAIN_COLLECTION = 'flickr8kzhbJanbosontrain'
DEFAULT_VAL_COLLECTION = 'flickr8kzhbJanbosonval'
DEFAULT_TEST_COLLECTION = 'flickr8kzhmbosontest'
DEFAULT_VISUAL_FEAT = 'pyresnet152-pool5osl2'
DEFAULT_MODEL_NAME = '8k_neuraltalk'
DEFAULT_BEAM_SIZE = 10
DEFAULT_FLUENCY_METHOD = 'sample' 
