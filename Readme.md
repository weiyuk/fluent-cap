
# Fluency-Guided Cross-Lingual Image Captioning

## Introduction

This is the code for the paper
Weiyu Lan, Xirong Li, Jianfeng Dong, [Fluency-Guided Cross-Lingual Image Captioning](https://arxiv.org/abs/1708.04390), ACM MM 2017 .

In this paper, we present an approach to cross-lingual image captioning by utilizing machine translation. A fluency-guided learning framework is proposed to deal with the lack of fluency in machine-translated sentences.
This repository provides data and code for training a Chinese captioning model that can generate fluent and relevant Chinese captions for a given image. 
With machine translated captions in other languages and estimated fluency scores, you can also train a fluency-guided captioning model for the new target language.


## Requirements
### Install Required Packages

First ensure that you have installed the following required packages:

* **TensorFlow** 1.0 or greater ([instructions](https://www.tensorflow.org/install/))

### Prepare the Data

Run `download_cn_data.sh` to get the text data and 
extracted feature from [ResNet-152](https://github.com/KaimingHe/deep-residual-networks) on flickr8k and flickr30k (totally ~296M).
Text data includes machine-translated Chinese captions, estimated fluency scores, 
and human-translated captions on test sets for evaluation.
Word segmentation is performed to tokenize a given sentence to a sequence of Chinese words
using [boson](http://bosonnlp.com/),
since Chinese sentences are written without explicit word delimiters. 

Extracted data is placed in `$HOME/VisualSearch/`.


## Training and Evaluating a Model

Run the script.

```shell
cd doit
bash do-all.sh
```

Running the script will do the following things:

1. Generate a dictionary on the training set, keeping words that occur >= 5 times
2. Train the fluency-guided cross-lingual image captioning model using rejection sampling and dump the model checkpoints
3. Run evaluation on the validation set and log loss information of the checkpoints
4. Generate captions on test set using the checkpoint that perform best on the validation set and evaluate the performance

The trained model and the evaluation results are all shown in `$HOME/VisualSearch/$collection/`

## Expected Performance

The expected performance of different fluency-guided approaches on Flickr8k-cn is as follows:

| Approach | BLEU4 | ROUGE_L | CIDEr |
| ------------- | ------------- | ------------- | ------------- |
| Without fluency | 24.1  | 45.9  | 47.6  |
| Fluency-only | 20.7  | 41.1  | 35.2  |
| Rejection sampling | 23.9  | 45.3  | 46.6  |
| Weighted loss | 24.0  | 45.0  | 46.3  |


