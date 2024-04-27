# XTuner 微调 Llama3 个人小助手认知
## 1 安装lesson1完成Web版部署 【已完成】
## 2 自我认知训练数据集准备
#### 2.1 生成数据文件
<img src=".\imgs\ls2-2-1.png">`
```shell
cd ~/Llama3-XTuner-CN
python tools/gdata.py 
```
###### 生成的数据位置在  ~/Llama3-XTuner-CN/data/personal_assistant.json
<img src=".\imgs\ls2-1.png">

####  2.2 XTuner配置文件准备
#### 2.3 训练模型
```shell
------cd /root/project/llama3-ft----- 错误路径,已经创建路径需要删除
cd /root/Llama3-XTuner-CN/
mkdir -p /root/llama3_pth

# 开始训练,使用 deepspeed 加速，A100 40G显存 耗时24分钟,A100 24G显存 耗时19分钟
xtuner train configs/assistant/llama3_8b_instruct_qlora_assistant.py --work-dir /root/llama3_pth

# Adapter PTH 转 HF 格式  9：28开始,9:36结束 ，耗时 8分钟
xtuner convert pth_to_hf /root/llama3_pth/llama3_8b_instruct_qlora_assistant.py \
  /root/llama3_pth/iter_500.pth \
  /root/llama3_hf_adapter

# 模型合并
export MKL_SERVICE_FORCE_INTEL=1
xtuner convert merge /root/model/Meta-Llama-3-8B-Instruct \
  /root/llama3_hf_adapter\
  /root/llama3_hf_merged
```
##### 2.3.1 训练前脚本修改
<img src=".\imgs\ls2-2.png">

##### 2.3.2 训练中,训练开始时间 9：08，训练结束时间：9:27 ,耗时 19分钟
<img src=".\imgs\ls2-3.png">

##### 2.3.3 训练完成
<img src=".\imgs\ls2-4.png">

#### 2.4 推理验证
```shell
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  /root/llama3_hf_merged
```
##### 2.4.1启动最新的模型，开启问答模式
<img src=".\imgs\ls2-5.png">

##### 2.4.2 变傻子了
<img src=".\imgs\ls2-6.png">

