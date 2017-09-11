import os
from constant import ROOT_PATH, DEFAULT_WORD_COUNT, DEFAULT_LANG

def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']


def get_sent_file(collection, language=DEFAULT_LANG, rootpath=ROOT_PATH):
    if 0 == language:
        sent_file = os.path.join(rootpath, collection, 'TextData',  "%s.caption.txt" % collection)
    else:
        sent_file = os.path.join(rootpath, collection, 'TextData',  "seg.%s.caption.txt" % collection)
    return sent_file

def get_sent_score_file(collection, language=DEFAULT_LANG, rootpath=ROOT_PATH):
    if 0 == language:
        sent_score_file = None
    else:
        sent_score_file = os.path.join(rootpath, collection, 'TextData',  "%s.sent_score.txt" % collection)
    return sent_score_file

def get_vocab_name(word_cnt_thr=DEFAULT_WORD_COUNT):
    return 'vocab_count_thr_%d' % word_cnt_thr

def get_vocab_file(collection, word_cnt_thr=DEFAULT_WORD_COUNT, rootpath=ROOT_PATH):
    return os.path.join(rootpath, collection, 'TextData', 'vocab', '%s.txt' % get_vocab_name(word_cnt_thr))

def get_train_vocab_file(FLAGS):
    return get_vocab_file(FLAGS.train_collection, FLAGS.word_cnt_thr, FLAGS.rootpath)

def get_fluency_method_name(fluency_method):
    if fluency_method == None:
        return 'without_fluency'
    else:
        return fluency_method

def get_model_dir(FLAGS):
    return os.path.join(FLAGS.rootpath, FLAGS.train_collection, 'Models', get_fluency_method_name(FLAGS.fluency_method), FLAGS.model_name, get_vocab_name(FLAGS.word_cnt_thr), FLAGS.vf_name)

def get_variable_dir(FLAGS):
    return os.path.join(get_model_dir(FLAGS), 'variables')

def get_pred_dir(FLAGS):
    output = 'autocap'
    return os.path.join(FLAGS.rootpath, FLAGS.test_collection, output, FLAGS.test_collection, FLAGS.train_collection, get_fluency_method_name(FLAGS.fluency_method), FLAGS.model_name, get_vocab_name(FLAGS.word_cnt_thr), FLAGS.vf_name, 'bs%d' % FLAGS.beam_size) 

def get_sim_dir(FLAGS):
    return os.path.join(FLAGS.rootpath, FLAGS.val_collection, 'SimilarityIndex', FLAGS.val_collection, FLAGS.train_collection, get_fluency_method_name(FLAGS.fluency_method), FLAGS.model_name,
                        get_vocab_name(FLAGS.word_cnt_thr), FLAGS.vf_name)

def get_feat_dir(collection, feature, rootpath):
    return os.path.join(rootpath, collection, 'FeatureData', feature)

def get_train_feat_dir(FLAGS):
    return get_feat_dir(FLAGS.train_collection, FLAGS.vf_name, FLAGS.rootpath)

def get_val_feat_dir(FLAGS):
    return get_feat_dir(FLAGS.val_collection, FLAGS.vf_name, FLAGS.rootpath)

def get_test_feat_dir(FLAGS):
    return get_feat_dir(FLAGS.test_collection, FLAGS.vf_name, FLAGS.rootpath)

def get_metrics(lang):
    if 0 == lang: # English
        metrics = str.split('Bleu_4 METEOR ROUGE_L CIDEr')
    else: # Chinese does not support METEOR
        metrics = str.split('Bleu_4 ROUGE_L CIDEr')
    return metrics

