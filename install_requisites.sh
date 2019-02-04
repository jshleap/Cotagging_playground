#!/usr/bin/env bash

cwd=$PWD

# try to keep setuptools up-to-date
python3 -m pip install --upgrade pip setuptools wheel --user
# install dask
python3 -m pip install "dask[complete]" --user
python3 -m pip install pandas --user
python3 -m pip install scipy --user
python3 -m pip install chest --user
python3 -m pip install dill --user
# install flashpca locally
mkdir -p software && cd software
wget https://github.com/gabraham/flashpca/releases/download/v2.0/flashpca_x86-64.gz
gunzip flashpca_x86-64.gz
chmod +x flashpca_x86-64
# get flashpca in your profile
echo "export PATH=$PATH:$PWD" >> ~/.bashrc
# install pandas_plink
git clone https://github.com/limix/pandas-plink.git && cd pandas-plink
python3 setup.py install --prefix=$HOME/.local
cd ${cwd}