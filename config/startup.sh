#!/bin/bash

cp .vimrc ~/.vimrc

ipython profile create
cp ipython_config.py ~/.ipython/profile_default/ipython_config.py

cp -a @jupyterlab ~/.jupyter/lab/user-settings
