source common_zh.ini

top_k=1
beam_size=5
length_normalization_factor=0.5

python ../test_models.py --train_collection $train_collection --val_collection $val_collection --test_collection $test_collection --model_name $model_name --vf_name $vf --top_k $top_k --length_normalization_factor $length_normalization_factor --beam_size $beam_size --overwrite $overwrite--fluency_method $fluency_method 



