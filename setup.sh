sudo apt-get update && sudo apt-get -f -y install espeak python3-pip unzip git wget gunicorn python3.10-venv
python3.10 -m venv env
source env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
git clone https://github.com/jaywalnut310/vits
cd vits
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gdown
gdown 11aHOlhnxzjpdWDpsz1vFDCzbeEfoIxru

mv ../setup_vits.py setup.py

mv monotonic_align ../monotonic_align
cd ../monotonic_align
pip install Cython
python setup.py build_ext --inplace
cd ..

mkdir "VSTs"
cd VSTs
wget https://tal-software.com/downloads/plugins/Tal-Vocoder-2_64_linux-2.2.1.zip
unzip Tal-Vocoder-2_64_linux-2.2.1.zip
mv TAL-Vocoder-2/* .
rm Tal-Vocoder-2_64_linux-2.2.1.zip

cd ..
pip install -r requirements.txt

cd vits
pip install -e .
cd ..
