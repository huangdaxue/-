## 1 LMDeploy环境部署，
<img src =".\imgs\hw5-1.png">

<img src =".\imgs\hw5-2.png">

<img src =".\imgs\hw5-3.png">

## 量化设置

#### KV8量化以及GPU占用情况
```shell
lmdeploy chat /root/internlm2-chat-1_8b
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.5
lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.01
```
<img src =".\imgs\hw5-4.1.png">
<img src =".\imgs\hw5-4.2.png">
<img src =".\imgs\hw5-4.3.png">

#### W4A16量化及其现存占用
```shell
lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.01
```
<img src =".\imgs\hw5-4.4.png">


#### LMDeploy服务(serve)
##### 4.1 启动API服务器
<img src =".\imgs\hw5-5.png">

#### 4.2 命令行客户端连接API服务器
<img src =".\imgs\hw5-6.png">
#### 4.3 网页客户端连接API服务器
<img src =".\imgs\hw5-7.png">

#### Python代码集成
##### Python代码集成运行1.8B模型
<img src =".\imgs\hw5-8.png">

#####  向TurboMind后端传递参数
<img src =".\imgs\hw5-9.png">

#### 拓展部分
##### 第一图使用LMDeploy运行视觉多模态大模型llava,第二图 Gradio来运行llava模型
<img src =".\imgs\hw5-10.png">
<img src =".\imgs\hw5-11.png">

#### 使用LMDeploy运行第三方大模型
##### 1 Transformer库的推理速度
<img src =".\imgs\hw5-12.png">

##### 2 LMDeploy的推理速度
<img src =".\imgs\hw5-13.png">
