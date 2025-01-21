#!/bin/sh

# create virtual environment:
python -m venv venv
source venv/bin/activate

# install requirements:
pip install -r ./requirements.txt

# download bart score:
wget -P ./resources/ https://raw.githubusercontent.com/neulab/BARTScore/main/bart_score.py

mkdir ./data