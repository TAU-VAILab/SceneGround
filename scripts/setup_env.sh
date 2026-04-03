set -e
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install -e ./nerfstudio

pip install -r ./requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

pip install git+https://github.com/nerfstudio-project/gsplat.git@5fc940b --no-build-isolation

# Fix numpy version compatibility
pip uninstall numpy
pip install numpy==1.26.4

pip install timm==1.0.7
