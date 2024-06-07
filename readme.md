# SCIR-SC组 多模态小组demo展示源代码
## face-demo环境配置
自行创建python环境，配置gradio
运行方式：python chat_demo3.py
## qwenvl环境配置
### 环境要求
新创建一个环境，配置下述库
transformers==4.32.0
accelerate
tiktoken
einops
transformers_stream_generator==0.0.4
scipy
torchvision
pillow
tensorboard
matplotlib
### 下载微调模型
链接：https://pan.baidu.com/s/1Z2AdOI4qI-80iE9ky0oEcA 
提取码：scir
## Sadtalker前端通信代码
将sadtalker_demo.py文件放入到SadTalker文件目录中，配置好相应的环境即可
## 项目启动方法
首先启动后端的三个部分：对话情感模型，text2audio模型，audio2video模型；
对话情感模型启动方法：<code>uvicorn qwenvl_chat:app --server "0.0.0.0" --host 9990</code>;同理方法启动另外两个模型，避免端口号冲突，修改host参数。demo如果部署在本地，则需要通过ssh反向代理来进行，ssh -N -L 22221:gpuxx:9990 hpc。
