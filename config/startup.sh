#!/bin/bash

# run this script from project root in main shell, i.e. with: . ./startup.sh

# export settings
ipython profile create
cp ipython_config.py ~/.ipython/profile_default/ipython_config.py
cp -a @jupyterlab ~/.jupyter/lab/user-settings
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
. ~/.bashrc
# conda init bash
conda env create -f environment.yaml
conda activate nysparcs
