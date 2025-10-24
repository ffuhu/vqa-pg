#!/bin/bash

set -e  # Exit on error

apt update
apt install screen

mkdir /root/.uvenvs
cd /root/.uvenvs

uv venv vqa-pg --python 3.11
source vqa-pg/bin/activate

cd /root/vqa-pg
uv pip install -r requirements.txt

echo "Login into wandb and huggingface!"

wandb login