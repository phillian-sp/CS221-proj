#!/bin/bash

# Install wget and bzip2 (needed for Miniconda)
apt-get update
apt-get install -y wget bzip2 git

# Download and install Miniconda
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
wget -q $MINICONDA_URL -O miniconda.sh
bash miniconda.sh -b -p /opt/conda
rm miniconda.sh

# Add conda to path
export PATH="/opt/conda/bin:$PATH"

# Clone repository (replace with your repo)
git clone <your-repo-url>
cd <repo-directory>

# Create conda environment from file
conda env create -f environment.yml

# Activate conda environment
source /opt/conda/bin/activate forecasting

# Make conda environment available to all users
echo "export PATH=/opt/conda/bin:$PATH" >> /etc/profile
echo "source activate forecasting" >> /etc/profile
