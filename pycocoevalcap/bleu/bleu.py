#!/usr/bin/env python
# 
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>
import numpy as np
from bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, n=4):
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):
        #assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            bleu_scorer += (hypo[0], ref)

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=1)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        return score, scores

    def method(self):
        return "Bleu"

    def compute_score_for_consensus(self, gts):

        assert(len(gts.keys()) == 1)
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            ref = gts[id]

            sent_len = len(ref)
            
            # Sanity check.
            #assert(type(hypo) is list)
            #assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 1)

            for r1 in ref:
                for r2 in ref:
                    bleu_scorer += (r1, [r2])

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)

        scores = np.sum(scores, axis = 0)
        scores = scores.reshape((sent_len, sent_len))
        scores = np.sum(scores, axis = 1)        

        return score, scores


    def compute_score_for_consensus2(self, gts, subGts):

        assert(len(gts.keys()) == 1)
        assert(gts.keys() == subGts.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            ref = gts[id]
            subRef = subGts[id]

            sent_len = len(ref)

            # Sanity check.
            #assert(type(hypo) is list)
            #assert(len(hypo) == 1)
            assert(type(subRef) is list)
            assert(type(ref) is list)
            assert(len(ref) > 1)

            for r1 in ref:
                for r2 in subRef:
                    bleu_scorer += (r1, [r2])

        #score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        #score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        scores = np.sum(scores, axis = 0)
        scores = scores.reshape((len(ref), len(subRef)))
        scores = np.sum(scores, axis = 1)

        return score, scores



