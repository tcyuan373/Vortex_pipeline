conda create -n FLMR python=3.12 -y
conda activate FLMR

#faiss gpu
conda install -c pytorch/label/nightly -c nvidia faiss-gpu
#torch and dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ujson gitpython easydict ninja datasets transformers

#FLMR
git clone https://github.com/LinWeizheDragon/FLMR.git
cd FLMR
pip install -e .
#ColBERT
cd third_party/ColBERT
pip install -e .

