from __future__ import print_function, division
import sys, os, re
import numpy as np
import codecs
import logging

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from constant import ROOT_PATH, DEFAULT_LANG
from text import TextTool
import utility

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

'''
# process sentence for performance evaluation
def process_sent(sent):
    # delete non-ascii chars
    sent = sent.decode('utf-8').encode('ascii','ignore')
    sent = sent.replace('\r',' ')
    sent = re.sub(r"[^A-Za-z0-9]", " ", sent).lower()
    sent = ' '.join([x for x in sent.split() if x])
    return sent
'''

class EvalCap:
    def __init__(self, ground_truth_fname, lang=DEFAULT_LANG):
        self.eval = {}
        self.imgToEval = {}
        self.gts = {}
        
        data = open(ground_truth_fname).readlines() if 0 == lang else codecs.open(ground_truth_fname, 'r', 'utf-8').readlines() 
        for line in data:
            sent_id, sent = line.strip().split(' ', 1)
            sent = ' '.join(TextTool.tokenize(sent, lang)) #process_sent(sent)
            img_id = os.path.splitext(sent_id.split('#')[0])[0]
            self.gts.setdefault(img_id, []).append(sent)

        logger.info('setting up scorers...')
        if 0 == lang:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]


    def evaluate(self, res):
        imgIds = self.gts.keys()
        result = []

        for scorer, method in self.scorers:
            logger.debug('computing %s score...', scorer.method())
            score, scores = scorer.compute_score(self.gts, res)

            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    logger.debug("%s: %0.3f", m, sc)
                    result.append("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                logger.debug("%s: %0.3f", method, score)
                result.append("%s: %0.3f" % (method, score))

        return result

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def getEvalImgs(self):
        return [eval for imgId, eval in self.imgToEval.items()]


def read_preds(pred_fname, lang=DEFAULT_LANG):
    res = {}
    for line in codecs.open(pred_fname, 'r', 'utf-8').readlines():
        name, score, sent = line.split(' ',2)
        #res[name] = [process_sent(sent)]
        res[name] = [' '.join(TextTool.tokenize(sent, lang))]
    return res


def batch_eval(eval_cap, pred_fnames, lang=DEFAULT_LANG, metrics=['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']):
    n_runs = len(pred_fnames)
    n_metrics = len(metrics)
    perf_table = np.zeros((n_runs, n_metrics))

    for i,pred_fname in enumerate(pred_fnames):
        res = read_preds(pred_fname, lang)
        results = eval_cap.evaluate(res)
        perf_table[i,:] = [eval_cap.eval[method] for method in metrics]
    return perf_table
      
def print_performance_table(perf_table, run_names=None):
    n_runs, n_metrics = perf_table.shape
    if not run_names:
        run_names = ['run%d'%i for i in range(n_runs)]

    for i in range(n_runs):
        print( run_names[i], ' '.join(map(lambda x: '%.1f'%(x*100), perf_table[i,:])), '%.1f'% (100*perf_table[i,:].sum()) )


def get_run_name(pred_fname):
    p = re.compile(r'autocap/[\w]+/(?P<name>.+)')
    return p.search(pred_fname).group('name')


def dry_run():
    from constant import ROOT_PATH as rootpath
    testCollection = 'flickr8kenctest'
    runfile = os.path.join(rootpath, testCollection, 'runs.txt')
    metrics = str.split('Bleu_4 METEOR ROUGE_L CIDEr')
    pred_files = [x.strip() for x in open(runfile).readlines() if x.strip() and not x.strip().startswith('#')]
    pred_files.sort()
    ground_truth_fname = os.path.join(rootpath, testCollection, 'TextData', '%s.caption.txt' % testCollection)
    eval_cap = EvalCap(ground_truth_fname)

    perf_table = batch_eval(eval_cap, pred_files)
    print (' '.join(metrics + ['SUM']))
    print_performance_table(perf_table, [get_run_name(x) for x in pred_files])


def process(options, collection, pred_file):
    rootpath = options.rootpath
    lang = options.language

    metrics = utility.get_metrics(lang)
    pred_files = [pred_file] if options.is_filelist == 0 else [x.strip() for x in open(pred_file).readlines() if x.strip() and not x.strip().startswith('#')]
    gt_file = utility.get_sent_file(collection, lang, rootpath)
    eval_cap = EvalCap(gt_file, lang)
    perf_table = batch_eval(eval_cap, pred_files, lang, metrics=metrics)
    print (' '.join(metrics + ['SUM']))
    print_performance_table(perf_table, [get_run_name(x) for x in pred_files])     

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] test_collection pred_file""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--language", default=DEFAULT_LANG, type="int", help="language, 0: English, 1: Chinese (default: %d)" % DEFAULT_LANG)
    parser.add_option("--is_filelist", default=0, type="int", help="each line of pred_file is a prediction filename (default: 0)")

    (options, args) = parser.parse_args(argv)

    if len(args) < 2:
        parser.print_help()
        return 1
    
    return process(options, args[0], args[1])

if __name__ == '__main__':
   sys.exit(main())  
