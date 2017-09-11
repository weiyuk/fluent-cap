# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import codecs
import re
import logging

from constant import ROOT_PATH, DEFAULT_LANG, TOKEN_UNK, DEFAULT_WORD_COUNT

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)



if 3 == sys.version_info[0]:
    CHN_DEL_SET = '， 。 、 ！ 《 》 “ ” ； ？ ‘ ’ '.split()
else:
    CHN_DEL_SET = [x.decode('utf-8') for x in '， 。 、 ！ 《 》 “ ” ； ？ ‘ ’ '.split()]


class TextTool:

    @staticmethod
    def tokenize(input_str, language=DEFAULT_LANG):
        if 0 == language: # English
            # delete non-ascii chars
            #sent = input_str.decode('utf-8').encode('ascii', 'ignore')
            sent = input_str
            sent = sent.replace('\r',' ')
            sent = re.sub(r"[^A-Za-z0-9]", " ", sent).strip().lower() 
            tokens = sent.split()
        else: # Chinese  
            sent = input_str #string.decode('utf-8')
            for elem in CHN_DEL_SET:
                sent = sent.replace(elem,'')
            #sent = sent.encode('utf-8')
            sent = re.sub("[A-Za-z]", "", sent)
            tokens = [x.split(':')[0] for x in sent.split()] # use split(':')[0] because each word might be followed by its POS tag

        return tokens


class TextBank:
    def __init__(self, vocab_file):
        """Initialize vocabulary from file."""
        assert os.path.exists(vocab_file), 'File does not exists %s' % vocab_file
        with codecs.open(vocab_file,'r','utf-8') as f:
            self.rev_vocab = [x.strip() for x in f.readlines()]
            f.close()
        self.vocab = dict([(x, y) for (y, x) in enumerate(self.rev_vocab)])
        assert TOKEN_UNK in self.vocab
        logger.info('load %d words from %s', len(self.vocab), vocab_file)
 

    def encode_tokens(self, tokens, flag_add_bos=False):
        """Encode individual tokens in a sentence with their index in vocab."""
        encoded_tokens = [self.vocab.get(word, self.vocab[TOKEN_UNK]) for word in tokens]
        if flag_add_bos:
          assert '<bos>' in self.vocab
          encoded_tokens = [self.vocab['<bos>']] + encoded_tokens + [self.vocab['<bos>']]
        return encoded_tokens

    def decode_tokens(self, encoded_tokens, flag_remove_bos=True):
        """Decode words index of a sentence to words."""
        if flag_remove_bos and encoded_tokens[-1] == self.vocab['<bos>']:
            words = [self.rev_vocab[x] for x in encoded_tokens[:-1]]
        else:
            words = [self.rev_vocab[x] for x in encoded_tokens]
        return words

if __name__ == '__main__':
    import utility
    rootpath = ROOT_PATH
    collection = 'flickr8kenctrain'
    lang = 0
    sent_file = utility.get_sent_file(collection, lang)
    word_cnt_thr = 5
    vocab_file = utility.get_vocab_file(collection)
    textbank = TextBank(vocab_file)
    fr = codecs.open(sent_file,'r','utf-8')
    for line_index, line in enumerate(fr):
        sent_id, sent = line.strip().split(" ", 1)
        print (sent)
        tokens = TextTool.tokenize(sent, lang)
        print (' '.join(tokens))
        encoded_tokens =  textbank.encode_tokens(tokens)
        print (encoded_tokens)
        words = textbank.decode_tokens(encoded_tokens)
        print (' '.join(words))
        break
    fr.close()

