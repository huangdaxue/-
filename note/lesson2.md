## 0 创建开发机，最低配置即可，选择cuda11.7
<img src =".\imgs\ls2-01.png" >

## 1 下载并部署 InternLM2-Chat-1.8B 模型进行智能对话
#### 1 下载代码
```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```

#### 2 部署代码
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```
#### 3 请创作一篇500字的鬼故事
<img src =".\imgs\ls2-02.png" >

## 2 部署实战营优秀作品 八戒-Chat-1.8B 模型
#### 1 下载代码
```python
cd /root/
git clone https://gitee.com/InternLM/Tutorial -b camp2
# git clone https://github.com/InternLM/Tutorial -b camp2
cd /root/Tutorial
python /root/Tutorial/helloworld/bajie_download.py
```
#### 2 执行代码
```python
streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
```
<img src =".\imgs\ls2-03.png" >

3 #### web界面访问
<img src =".\imgs\ls2-04.png" >

## 3 使用 Lagent 运行 InternLM2-Chat-7B 模型
#### 1 拉去代码并安装
```shell
conda activate demo
cd /root/demo
git clone https://gitee.com/internlm/lagent.git
cd /root/demo/lagent
git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
pip install -e .
```
#### 2 创建 Internlm2-chat-7b 的模型软链接到 /root/models/internlm2-chat-7b
```shell
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
```
#### 3 将模型路径改为 /root/models/internlm2-chat-7b
修改文件  /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py
<img src =".\imgs\hw2-4.png">
<img src ="..\homework\imgs\hw2-5.png">

#### 4 启动 Web demo 并将 6006 端口映射到本地，然后在本地浏览器打开 demo 项目
```shell
streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006
```

## 4 浦语·灵笔2 的 图文创作步骤
#### 1 切换到 conda demo 环境并安装必须的补充包
```shell
conda activate demo
pip install timm==0.4.12 sentencepiece==0.1.99 markdown2==2.4.10 xlsxwriter==3.1.2 gradio==4.13.0 modelscope==1.9.5
```
#### 2 克隆 InternLM-XComposer 仓库 并切换到指定提交
```shell
cd /root/demo
git clone https://gitee.com/internlm/InternLM-XComposer.git
cd /root/demo/InternLM-XComposer
git checkout f31220eddca2cf6246ee2ddf8e375a40457ff626
```
#### 3 创建 浦语·灵笔2 模型软链接
```shell
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b /root/models/internlm-xcomposer2-7b
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b /root/models/internlm-xcomposer2-vl-7b
```
#### 4 启动图文写作 Demo 并转发 6006 端口到本地，然后在本地浏览器打开 http://127.0.0.1:6006
```shell
cd /root/demo/InternLM-XComposer
python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
--code_path /root/models/internlm-xcomposer2-7b \
--private \
--num_gpus 1 \
--port 6006
```

