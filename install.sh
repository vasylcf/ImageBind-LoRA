#!/bin/bash

sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt-get update && sudo apt-get install google-cloud-cli

gcloud auth login --cred-file /home/coder/project/motion-tracking.json
export GOOGLE_APPLICATION_CREDENTIALS=/home/coder/project/motion-tracking.json
gsutil cp gs://motion_tracking_data/lcf/ted/TED_chunks_DS.zip .

sudo apt install libgeos-dev  
pip install -r requirements.txt    
export PATH="$PATH:/home/coder/.local/bin"

pip install fastapi
pip install umap-learn==0.5.3
pip install seaborn
pip install  distinctipy==1.2.2
pip install bokeh