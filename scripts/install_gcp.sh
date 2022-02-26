sudo ln -s /usr/bin/python3.8 /usr/bin/python
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
python scripts/download_flownet2.py