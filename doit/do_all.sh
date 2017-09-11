source common_zh.ini

collections=( $train_collection $val_collection $test_collection )
for collection in "${collections[@]}"
do
    feat_dir=$rootpath/$collection/FeatureData/$vf
    if [ ! -d ${feat_dir} ]; then
        echo "${feat_dir} not found"
        exit
    fi
done

# --------------------------------------------
# 1. training
# --------------------------------------------

python ../generate_vocab.py $train_collection --language $lang --rootpath $rootpath
python ../trainer.py --model_name $model_name  --train_collection $train_collection --language $lang --vf_name $vf --overwrite $overwrite --rootpath $rootpath --fluency_method $fluency_method --sent_score_file $train_sent_score_file

# --------------------------------------------
# 2. validation
# --------------------------------------------

python ../compute_val_loss.py --train_collection $train_collection --val_collection $val_collection --model_name $model_name --vf_name $vf --language $lang --overwrite $overwrite --rootpath $rootpath --fluency_method $fluency_method --sent_score_file $val_sent_score_file

# --------------------------------------------
# 3. test
# --------------------------------------------

top_k=1
beam_size=5
#beam_size=20
#beam_size=10
length_normalization_factor=0.5

python ../test_models.py --train_collection $train_collection --val_collection $val_collection --test_collection $test_collection --model_name $model_name --vf_name $vf --top_k $top_k --length_normalization_factor $length_normalization_factor --beam_size $beam_size --overwrite $overwrite  --rootpath $rootpath --fluency_method $fluency_method


# --------------------------------------------
# 4. evaluation
# --------------------------------------------

run_file=$rootpath/$test_collection/autocap/runs.txt
find $rootpath/$test_collection/autocap/$test_collection | grep top0/top_one_pred_sent.txt > ${run_file}

if [ ! -f ${run_file} ]; then
    echo "${run_file} not found"
    exit
fi

python ../eval.py $test_collection $run_file --is_filelist 1 --language $lang

