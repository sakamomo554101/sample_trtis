#!/usr/bin/env bash

# download mecab archive
wget -O mecab-0.996.tar.gz "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7cENtOXlicTFaRUE"
tar -xzvf mecab-0.996.tar.gz
rm mecab-0.996.tar.gz

# install mecab
cd mecab-0.996
./configure
make
make install
make check
cd ../

# download ipadic
wget -O mecab-ipadic-2.7.0-20070801.tar.gz "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM"
tar zxvf mecab-ipadic-2.7.0-20070801.tar.gz
rm mecab-ipadic-2.7.0-20070801.tar.gz

# install ipadic
cd mecab-ipadic-2.7.0-20070801
./configure --with-charset=utf8
ldconfig
make
make install
