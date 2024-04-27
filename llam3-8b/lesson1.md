# 实践教程（InternStudio 版）
### 1 环境配置
```shell
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

```
### 2 下载模型
##### 2.1 安装 git-lfs 依赖
```shell
conda install git
conda  install git-lfs
```
##### 2.2 下载模型
```shell
mkdir -p ~/model
cd ~/model
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```
##### 2.2 或者软链接 InternStudio 中的模型
```shell
ln -s /root/share/new_models/meta-llama/Meta-Llama-3-8B-Instruct ~/model/Meta-Llama-3-8B-Instruct
```
### 3 Web Demo 部署
##### 3.1 下载Llama3-XTuner-CN
```shell
cd ~
git clone https://github.com/SmartFlowAI/Llama3-XTuner-CN
```
<img src =".\imgs\ls1-1.png">

##### 3.2 安装 XTuner 时会自动安装其他依赖
```shell
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .
```
##### 3.3 运行 web_demo.py
```shell
streamlit run ~/Llama3-XTuner-CN/tools/internstudio_web_demo.py \
  /root/model/Meta-Llama-3-8B-Instruct
```
```shell
streamlit run /root/Llama3-XTuner-CN/tools/internstudio_web_demo.py  /root/model/Meta-Llama-3-8B-Instruct
```
##### 3.4 问题演示
<img src =".\imgs\ls1-2.png">

<img src =".\imgs\ls1-3.png">