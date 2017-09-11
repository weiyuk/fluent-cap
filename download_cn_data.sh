ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH
wget -O $ROOTPATH/flickr8k-cn.tar.gz http://lixirong.net/data/mm2017/flickr8k-cn.tar.gz
wget -O $ROOTPATH/flickr30k-cn.tar.gz http://lixirong.net/data/mm2017/flickr30k-cn.tar.gz
cd $ROOTPATH
tar zxvf flickr8k-cn.tar.gz
tar zxvf flickr30k-cn.tar.gz
rm flickr8k-cn.tar.gz
rm flickr30k-cn.tar.gz
cd -
