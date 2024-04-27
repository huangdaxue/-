# XTuner 微调 Llama3 图片理解多模态【未完成，执行异常，现存不够】
## 环境安装
##### conda环境安装
```shell
conda create -n llama3 python=3.10
conda activate llama3
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
##### Xtuner 安装
```shell
cd ~
git clone -b v0.1.18 https://github.com/InternLM/XTuner
cd XTuner
pip install -e .
```
## 模型准备
### 1 准备 Llama3 权重
##### 1.1 InternStudio
```shell
cd ~
ln -s /root/new_models/meta-llama/Meta-Llama-3-8B-Instruct .
```
##### 1.2 OpenXLab 上下载
```shell
cd ~
git lfs install
git clone https://code.openxlab.org.cn/MrCat/Llama-3-8B-Instruct.git Meta-Llama-3-8B-Instruct
```
### 2 准备 Visual Encoder 权重
##### 2.1 InternStudio
```shell
cd ~
ln -s /root/new_models/openai/clip-vit-large-patch14-336 .
```
##### 2.2 非 InternStudio
```shell
https://huggingface.co/openai/clip-vit-large-patch14-336 以进行下载
```
### 3 准备 Image Projector 权重
##### 3.1 InternStudio
```shell
cd ~
ln -s /root/new_models/xtuner/llama3-llava-iter_2181.pth .
```
##### 3.2 非 InternStudio
```text
相关权重可以访问：https://huggingface.co/xtuner/llava-llama-3-8b 以及 https://huggingface.co/xtuner/llava-llama-3-8b-v1_1 。（已经过微调，并非 Pretrain 阶段的 Image Projector）
```
## 数据准备
```shell
cd ~
git clone https://github.com/InternLM/tutorial -b camp2
python ~/tutorial/xtuner/llava/llava_data/repeat.py \
  -i ~/tutorial/xtuner/llava/llava_data/unique_data.json \
  -o ~/tutorial/xtuner/llava/llava_data/repeated_data.json \
  -n 200
```
## 微调过程
#### 1 训练启动
###### r如果整体拷贝出现异常，逐步拷贝单条命令执行
```shell
cd ~
git clone https://github.com/SmartFlowAI/Llama3-XTuner-CN
mkdir -p ~/project/llama3-ft
cd ~/project/llama3-ft
mkdir -p /root/project/llama3-ft/llava 
xtuner train ~/Llama3-XTuner-CN/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py --work_dir ~/project/llama3-ft/llava --deepspeed deepspeed_zero2

xtuner train /root/Llama3-XTuner-CN/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py --work_dir /root/project/llama3-ft/llava --deepspeed deepspeed_zero2
```
###### 训练过程所需显存约为44447 MiB，在单卡A100上训练所需时间为30分钟。在训练好之后，我们将原始 image projector 和 我们微调得到的 image projector 都转换为 HuggingFace 格式，为了下面的效果体验做准备
```shell
xtuner convert pth_to_hf ~/Llama3-XTuner-CN/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
  ~/llama3-llava-iter_2181.pth \
  ~/project/llama3-ft/llava/pretrain_iter_2181_hf

xtuner convert pth_to_hf ~/Llama3-XTuner-CN/configs/llava_llama3_8b_instruct_qlora_clip_vit_large_p14_336_lora_e1_finetune.py \
  ~/project/llama3-ft/llava/iter_1200.pth \
  ~/project/llama3-ft/llava/finetune_iter_1200_hf
```

