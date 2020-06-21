#!/bin/bash

# run this script in main shell, i.e. with: source ./startup.sh

# export settings
ipython profile create
cp ipython_config.py ~/.ipython/profile_default/ipython_config.py
cp -ar @jupyterlab ~/.jupyter/lab/user-settings
cp .vimrc ~/.vimrc

# build and activate conda environment
if [ -d ~/miniconda3 ]
then
    . ~/miniconda3/etc/profile.d/conda.sh
    echo "Initializing miniconda"
elif [ -d ~/anaconda3 ]
then
    . ~/anaconda3/etc/profile.d/conda.sh
    echo "Initializing anaconda"
else
    echo "Warning: No miniconda or anaconda installation found"
fi

conda env create --file environment.yaml
echo ". /home/ec2-user/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate nysparcs

# set git user
git config --global user.name 'Eli Cutler'
git config --global user.email cutler.eli@gmail.com
