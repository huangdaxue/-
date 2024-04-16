## 1 LMDeploy环境部署，
#### 1.1创建conda环境
```shell
studio-conda -t lmdeploy -o pytorch-2.1.2
```
#### 1.2  安装LMDeploy
```shell
conda activate lmdeploy

pip install lmdeploy[all]==0.3.0
```
## 2.LMDeploy模型对话(chat)
#### 2.1 Huggingface与TurboMind
#### 2.2 下载模型
```shell
ls /root/share/new_models/Shanghai_AI_Laboratory/

cd ~

ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
# cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
```

#### 2.3 使用Transformer库运行模型
##### 创建文件 touch /root/pipeline_transformer.py 
```shell
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

inp = "hello"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=[])
print("[OUTPUT]", response)

inp = "please provide three suggestions about time management"
print("[INPUT]", inp)
response, history = model.chat(tokenizer, inp, history=history)
print("[OUTPUT]", response)

```
##### 运行文件
```shell
conda activate lmdeploy
python /root/pipeline_transformer.py
```
#### 2.4 使用LMDeploy与模型对话
```shell
conda activate lmdeploy
lmdeploy chat /root/internlm2-chat-1_8b
```
#### 3.LMDeploy模型量化(lite)
###### 主要包括 KV8量化和W4A16量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。
##### 3.1 设置最大KV Cache缓存大小
```shell
lmdeploy chat /root/internlm2-chat-1_8b
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.5
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.01
```
##### 3.2 使用W4A16量化
```shell
pip install einops==0.7.0

lmdeploy lite auto_awq \
   /root/internlm2-chat-1_8b \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 1024 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir /root/internlm2-chat-1_8b-4bit
  
 lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq
 
 lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.01
 
```
#### 4.LMDeploy服务(serve)

##### 4.1 启动API服务器
```shell
lmdeploy serve api_server \
    /root/internlm2-chat-1_8b \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
#### 4.2 命令行客户端连接API服务器
```shell
conda activate lmdeploy
lmdeploy serve api_client http://localhost:23333

```
#### 4.3 网页客户端连接API服务器
```shell
conda activate lmdeploy

lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \

```
#### 5.Python代码集成
##### 5.1 Python代码集成运行1.8B模型
```shell
conda activate lmdeploy
touch /root/pipeline.py

from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```
##### 5.2 向TurboMind后端传递参数
```shell
touch /root/pipeline_kv.py

from lmdeploy import pipeline, TurbomindEngineConfig

# 调低 k/v cache内存占比调整为总显存的 20%
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2)

pipe = pipeline('/root/internlm2-chat-1_8b',
                backend_config=backend_config)
response = pipe(['Hi, pls intro yourself', '上海是'])
print(response)
```
#### 6拓展部分
##### 6.1 使用LMDeploy运行视觉多模态大模型llava
```shell
conda activate lmdeploy

pip install git+https://github.com/haotian-liu/LLaVA.git@4e2277a060da264c4f21b364c867cc622c945874

touch /root/pipeline_llava.py

from lmdeploy.vl import load_image
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
response = pipe(('describe this image', image))
print(response)

python /root/pipeline_llava.py
```
##### 6.2 Gradio来运行llava模型，图形操作界面
```shell
touch /root/gradio_llava.py

import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

def model(image, text):
    if image is None:
        return [(text, "请上传一张图片。")]
    else:
        response = pipe((text, image)).text
        return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo.launch()   

```
##### 6.3使用LMDeploy运行第三方大模型
##### 6.4 定量比较LMDeploy与Transformer库的推理速度差异
benchmark_transformer.py
```shell
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()

# warmup
inp = "hello"
for i in range(5):
    print("Warm up...[{}/5]".format(i+1))
    response, history = model.chat(tokenizer, inp, history=[])

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response, history = model.chat(tokenizer, inp, history=history)
    total_words += len(response)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```
benchmark_lmdeploy.py
```shell
import datetime
from lmdeploy import pipeline

pipe = pipeline('/root/internlm2-chat-1_8b')

# warmup
inp = "hello"
for i in range(5):
    print("Warm up...[{}/5]".format(i+1))
    response = pipe([inp])

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response = pipe([inp])
    total_words += len(response[0].text)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
```
