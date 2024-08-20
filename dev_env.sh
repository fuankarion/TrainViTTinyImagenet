conda create --name tiny_imagenet python=3.10
conda activate tiny_imagenet

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install tqdm
pip install einops
pip install torchmetrics
pip install accelerate