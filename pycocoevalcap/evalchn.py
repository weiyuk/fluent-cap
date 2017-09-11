from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider


class EvalChn:
    def __init__(self, groundTruth, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.gts = groundTruth
        self.res = res
        self.params = {'image_id': res.keys()}

    def evaluate(self,verbose=False):
        imgIds = self.params['image_id']
        # Set up scorers
        # =================================================
        #if verbose:
        #  print 'tokenization...'
        #tokenizer = PTBTokenizer()

        #gts  = tokenizer.tokenize(self.gts)
        #res = tokenizer.tokenize(self.res)
        gts={}
        for key in self.gts:
          for x in self.gts[key]:
            gts.setdefault(key,[]).append(x['caption'])
        res={}
        for key in self.res:
          for x in self.res[key]:
            res.setdefault(key,[]).append(x['caption'])

        # =================================================
        # Set up scorers
        # =================================================
        if verbose:
            print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        result = []
        for scorer, method in scorers:
            if verbose:
                print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    if verbose:
                        print "%s: %0.3f"%(m, sc)
                        #print "avg:",sum(scs)/len(scs)
                    result.append("%s: %0.3f" % (m, sc))
                    #result.append("avg: %0.3f" % (sum(scs)/len(scs)))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                if verbose:
                    print "%s: %0.3f"%(method, score)
                result.append("%s: %0.3f" % (method, score))
        self.setEvalImgs()
        return result

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

