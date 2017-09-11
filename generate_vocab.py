from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import string
import codecs
from collections import Counter

#from util.common_utils import CommonUtiler
from constant import ROOT_PATH, DEFAULT_WORD_COUNT, DEFAULT_LANG
from text import TextTool
import utility

formatter_log = "[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s"
logging.basicConfig(
    format = formatter_log,
    datefmt = '%d %b %H:%M:%S')

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def process(options, collection):
    rootpath = options.rootpath
    count_thr = options.count_thr
    overwrite = options.overwrite
    lang = options.language
    assert(lang in [0, 1])

    sent_file = utility.get_sent_file(collection, lang, rootpath)
    result_dir = os.path.join(rootpath, collection, 'TextData', 'vocab')
    vocab_file = os.path.join(result_dir, 'vocab_count_thr_%d.txt' % count_thr)
    all_vocab_file = os.path.join(result_dir, 'vocab.txt')

    if os.path.exists(vocab_file) and not overwrite:
        logger.info('%s exists. quit' % vocab_file)
        return
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # set log to be written in the file
    hdlr = logging.FileHandler(os.path.join(result_dir, 'log_count_thr_%d.txt' % count_thr))
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(logging.Formatter(formatter_log))
    logger.addHandler(hdlr)

    word_cnt = Counter()
    sentlen_cnt = Counter()

    fr = codecs.open(sent_file,'r','utf-8')

    for line_index, line in enumerate(fr):
        sent_id, sent = line.strip().split(" ", 1)
        tokens = TextTool.tokenize(sent, lang)

        if 0 == line_index:
            logger.debug('input: %s', line.strip())
            logger.debug('sid: %s ', sent_id)
            logger.debug('processed sentence: %s', ' '.join(tokens))
        
        sentlen_cnt[len(tokens)] += 1
        for word in tokens:
            word_cnt[word] += 1

    word_count_list = word_cnt.most_common()
    vocab = [x[0] for x in word_count_list if x[1]>= count_thr]
    logger.info('number of unique words: %d', len(word_cnt))
    logger.info('size of the generated vocabulary: %d', len(vocab))
    

    with codecs.open(vocab_file, 'w','utf-8') as fw:
        fw.write('<pad>\n')
        fw.write('<unk>\n')
        fw.write('<bos>\n')
        fw.write('\n'.join(vocab))
        fw.close()

    with codecs.open(all_vocab_file, 'w','utf-8') as fw:
        fw.write('\n'.join(["%s %d" % (x[0],x[1]) for x in word_count_list]))
        fw.close()

    # Now lets look at the distribution of sentence length as well
    sentlen_count_list = sentlen_cnt.most_common()
    max_sentlen = max([x[0] for x in sentlen_count_list])
    logger.info('max sentence length: %d', max_sentlen)
    logger.info('sentlen count')
    sum_len = sum([x[1] for x in sentlen_count_list])
    assert ( (line_index + 1) == sum_len)

    for i in xrange(max_sentlen+1):
        if 0 == sentlen_cnt[i]:
            continue
        logger.debug('%2d: %10d   %f%%' % (i, sentlen_cnt[i], sentlen_cnt[i]*100.0/sum_len))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] collection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--count_thr", default=DEFAULT_WORD_COUNT, type="int", help="preserve words that have their occurrence no less than this threshold (default: %d)" % DEFAULT_WORD_COUNT)
    parser.add_option("--language", default=DEFAULT_LANG, type="int", help="language, 0: English, 1: Chinese (default: %d)" % DEFAULT_LANG)
    

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    return process(options, args[0])


if __name__ == "__main__":
    sys.exit(main())    

