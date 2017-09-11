source common.ini
source common_zh.ini

collections=( $train_collection $val_collection $test_collection )

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 feature"
    exit
fi

feature=$1

for collection in "${collections[@]}"
do
    feat_dir=$rootpath/${collection}/FeatureData/$feature
    if [ ! -d ${feat_dir} ]; then
        echo "${feat_dir} does not exist"
        exit
    fi
    python $HOME/github/jingwei/util/simpleknn/norm_feat.py $feat_dir
done


