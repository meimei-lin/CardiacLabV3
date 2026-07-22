## Install
'''bash
conda create --name CardiacLab python=3.9 -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
'''
## Run Server
### By cmd
'''bash
monailabel start_server \
--app radiology \
--studies "D:\tmp\data" \
--conf models segmentation_cardiac \
--conf network unetwic \
--conf --download_ckp_id <ID> \
--conf --target_spacing "0.7, 0.7, 1.0" \
--conf --spatial_size "128, 128, 128"\
--conf --intensity "-42, 423"
'''

## By script
set config to run/config.toml.
'''bash
model = 'unetwic'

[segmentation_cardiac.unetwic]
app = "radiology"
studies = "D:\\tmp\\data"
models = "segmentation_cardiac"
network = "unetwic"
download_ckp_id = "<ID>"
target_spacing = "0.7, 0.7, 1.0"
spatial_size = "128, 128, 128"
intensity = "-42, 423"
'''

run run/segmentation.py.
'''bash
python run/segmentation.py
'''
