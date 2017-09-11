class config(object):
  """Config for captioning model training."""
  # Training
  init_scale = 0.05 # variance of the gaussian variable initialization 
  max_grad_norm = 10 # gradient clipping
  learning_rate = 0.001 # inital learning rate (lr)
  lr_decay_keep = 80000 # Num. of iteration that we keep the initial lr
  lr_decay_iter = 4000 # Num. of iteration to apply lr_decay
  lr_decay = 0.85
  num_epoch = 100
  batch_size = 64
  num_epoch_save = 2 
  num_iter_verbose = 100 # Num. of iteration to print the training info.
  buckets = 20 # [6, 9, 12, 20]
  optimizer = 'adam'
  
  # Model parameter
  # keep_prob_mm = 0.5 # dropout rate of the multimodal layer, only for 'mrnn' type
  num_lstm_units = rnn_size = 512 # size of the rnn layer
  embedding_size = 512 # size of RNN cell and word emb
  # num_rnn_layers = 1
  max_num_steps = 20  # maxmimun length of training sentence
  lstm_dropout_keep_prob = keep_prob_rnn = 0.5 # dropout rate of the rnn output
  #keep_prob_emb = 0.5 # dropout rate of the word-embeddings

