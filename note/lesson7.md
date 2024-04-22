# OpenCompass ：是骡子是马，拉出来溜溜
## 1 面向GPU的环境安装
```shell
studio-conda -o internlm-base -t opencompass
source activate opencompass
mkdir -p /root/opencompass/
cd /root/opencompass/
git clone -b 0.2.4 https://github.com/open-compass/opencompass
cd opencompass
pip install -e .

pip install -r requirements.txt

cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip

```
## 2 查看支持的数据集和模型
```shell
python tools/list_configs.py internlm ceval

python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --num-gpus 1 --debug


```
