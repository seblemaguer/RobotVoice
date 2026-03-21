#!/bin/bash

# Install general python requirements
echo "\n\n"
echo "####################################################################"
echo "## Installing python dependencies"
echo "####################################################################"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gdown
pip install Cython

# Install VST Vocoder plugin
echo -e "\n\n"
echo "####################################################################"
echo "## Installing TAL Vocoder VST plugin"
echo "####################################################################"
(
    rm -rfv VSTs/
    mkdir "VSTs"
    cd VSTs
    wget https://tal-software.com/downloads/plugins/Tal-Vocoder-2_64_linux-2.2.1.zip
    unzip Tal-Vocoder-2_64_linux-2.2.1.zip
    mv TAL-Vocoder-2/* .
    rm Tal-Vocoder-2_64_linux-2.2.1.zip
)
