# LMDeploy 高效部署 Llama3 实践
## 1环境，模型准备
##### conda环境安装
```shell
# 如果你是InternStudio 可以直接使用
# studio-conda -t lmdeploy -o pytorch-2.1.2
# 初始化环境
conda create -n lmdeploy python=3.10
conda activate lmdeploy
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
##### 安装lmdeploy最新版
```shell
pip install -U lmdeploy[all]
```
##### Llama3 的下载
```shell
conda install git
git-lfs install

mkdir -p ~/model
cd ~/model
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct

#或者软链
mkdir -p ~/model
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct ~/model/Meta-Llama-3-8B-Instruct
```
## 2 LMDeploy chat
```shell
conda activate lmdeploy
lmdeploy chat /root/model/Meta-Llama-3-8B-Instruct
```
<img src=".\imgs\ls4-1.png">

## 3 LMDeploy模型量化(lite)
#### 3.1 设置最大KV Cache缓存大小
```shell
lmdeploy chat /root/model/Meta-Llama-3-8B-Instruct/
# 如果你是InternStudio 就使用
# studio-smi
nvidia-smi 

lmdeploy chat /root/model/Meta-Llama-3-8B-Instruct/ --cache-max-entry-count 0.5
# 如果你是InternStudio 就使用
# studio-smi
nvidia-smi 
#面来一波“极限”，把--cache-max-entry-count参数设置为0.01，约等于禁止KV Cache占用显存
lmdeploy chat /root/model/Meta-Llama-3-8B-Instruct/ --cache-max-entry-count 0.01
```
<img src=".\imgs\ls4-2-1.png">
<img src=".\imgs\ls4-2-2.png">

#### 3.2 使用W4A16量化
###### 仅需执行一条命令，就可以完成模型量化工作。
```shell
lmdeploy lite auto_awq \
   /root/model/Meta-Llama-3-8B-Instruct \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir /root/model/Meta-Llama-3-8B-Instruct_4bit
```
###### Chat功能运行W4A16量化后的模型
```shell
lmdeploy chat /root/model/Meta-Llama-3-8B-Instruct_4bit --model-format awq
```
###### KV Cache比例再次调为0.01，查看显存占用情况
```shell
lmdeploy chat /root/model/Meta-Llama-3-8B-Instruct_4bit --model-format awq --cache-max-entry-count 0.01
```
<img src=".\imgs\ls4-3.png">

###### 可见W4A16量化显存占用比量化前要低很多，约占四分之一

#### 3.3 在线量化 KV
###### 自 v0.4.0 起，LMDeploy KV 量化方式有原来的离线改为在线。并且，支持两种数值精度 int4、int8。量化方式为 per-head per-token 的非对称量化。它具备以下优势：
###### 量化不需要校准数据集
###### kv int8 量化精度几乎无损，kv int4 量化精度在可接受范围之内
###### 推理高效，在 llama2-7b 上加入 int8/int4 kv 量化，RPS 相较于 fp16 分别提升近 30% 和 40%
###### 支持 volta 架构（sm70）及以上的所有显卡型号：V100、20系列、T4、30系列、40系列、A10、A100 等等 通过 LMDeploy 应用 kv 量化非常简单，只需要设定 quant_policy 参数。LMDeploy 规定 qant_policy=4表示 kv int4 量化，quant_policy=8 表示 kv int8 量化。
## 4 LMDeploy服务(serve)
#### 4.1 启动API服务器
###### 通过以下命令启动API服务器，推理Meta-Llama-3-8B-Instruct模型：
```shell
lmdeploy serve api_server \
    /root/model/Meta-Llama-3-8B-Instruct \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
<img src=".\imgs\ls4-4.png">

#### 4.2 命令行客户端连接API服务器
###### 在“4.1”中，我们在终端里新开了一个API服务器。 本节中，我们要新建一个命令行客户端去连接API服务器。首先通过VS Code新建一个终端： 激活conda环境
```shell
conda activate lmdeploy

lmdeploy serve api_client http://localhost:23333
```
###### 执行效果
<img src=".\imgs\ls4-5.png">

#### 4.3 网页客户端连接API服务器
###### 新建一个VSCode终端，激活conda环境
```shell
conda activate lmdeploy
# 使用Gradio作为前端，启动网页客户端。
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```
<img src=".\imgs\ls4-6.png">

## 5. 推理速度
###### 使用 LMDeploy 在 A100（80G）推理 Llama3，每秒请求处理数（RPS）高达 25，是 vLLM 推理效率的 1.8+ 倍。
####  5.1 克隆仓库
```shell
cd ~
git clone https://github.com/Shengshenlan/Llama3-XTuner-CN.git
```
#### 5.2 下载测试数据
```shell
cd /root/lmdeploy
wget https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
#### 5.3 执行 benchmark 命令(如果你的显存较小，可以调低--cache-max-entry-count)
```shell
python benchmark/profile_throughput.py \
    ShareGPT_V3_unfiltered_cleaned_split.json \
    /root/model/Meta-Llama-3-8B-Instruct \
    -cache-max-entry-count 0.95 \
    --concurrency 256 \
    --model-format hf \
    --quant-policy 0 \
    --num-prompts 10000
```
#### 5.4 执行结果
```shell
concurrency: 256
elapsed_time: 399.739s

first token latency(s)(min, max, ave): 0.068, 4.066, 0.285
per-token latency(s) percentile(50, 75, 95, 99): [0, 0.094, 0.169, 0.227]

number of prompt tokens: 2238364
number of completion tokens: 2005448
token throughput (completion token): 5016.892 token/s
token throughput (prompt + completion token): 10616.453 token/s
RPS (request per second): 25.016 req/s
RPM (request per minute): 1500.979 req/min
```
## 6 使用LMDeploy运行视觉多模态大模型Llava-Llama-3
#### 6.1 安装依赖
```shell
pip install git+https://github.com/haotian-liu/LLaVA.git
```
#### 6.2 运行模型
###### 运行touch /root/pipeline_llava.py 新建一个文件夹，复制下列代码进去
```shell
touch /root/pipeline_llava.py

# 填充以下内容
from lmdeploy import pipeline, ChatTemplateConfig
from lmdeploy.vl import load_image
pipe = pipeline('xtuner/llava-llama-3-8b-v1_1-hf',
                chat_template_config=ChatTemplateConfig(model_name='llama3'))

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response.text)
```
```shell
cd /root/
python pipeline_llava.py
```
###### 运行结果

<img src=".\imgs\ls4-7.png">