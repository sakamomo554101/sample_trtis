#!/bin/bash
  
# build c++ library
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install
make -j16 trtis-clients

# update python library
cd ../
python -m pip install --upgrade install/python/tensorrtserver-*.whl
