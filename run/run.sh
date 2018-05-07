export http_proxy="http://10.223.133.20:52107"
export https_proxy="http://10.223.133.20:52107"
pip install --upgrade pip
pip install -r ../requirements.txt
#python pretrain.py
python train.py --model ./_out/default/model/epoch15
