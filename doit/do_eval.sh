source common.ini
#source common_zh.ini

#pred_file=/home/xirong/VisualSearch/$test_collection/autocap/$test_collection/flickr8kzhbJanbosontrain/8k_neuraltalk/vocab_count_thr_5/pygooglenet_bu4k-pool5_7x7_s1/top0/top_one_pred_sent.txt
run_file=$rootpath/$test_collection/autocap/runs.txt

find $rootpath/$test_collection/autocap/$test_collection | grep top0/top_one_pred_sent.txt > ${run_file}

if [ ! -f ${run_file} ]; then
    echo "${run_file} not found"
    exit
fi
#python eval.py $test_collection $pred_file --is_filelist 0 --language $lang
python ../eval.py $test_collection $run_file --is_filelist 1 --language $lang

