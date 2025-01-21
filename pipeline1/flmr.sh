conda create -n FLMR python=3.10 -y
conda activate FLMR


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/LinWeizheDragon/FLMR.git
cd FLMR
pip install -e .

cd third_party/ColBERT
pip install -e .

pip install ujson gitpython easydict ninja datasets transformers