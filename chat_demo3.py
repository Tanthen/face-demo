# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from pathlib import Path
import requests
import copy
import gradio as gr
import os
from PIL import Image
from io import BytesIO
import re
import secrets
import tempfile
import base64
import ipdb
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

url = 'http://localhost:22223/chat'
url_image = 'http://localhost:22221/getv'
share = False
inbrowser = False
server_port = 9933
server_name = '127.0.0.1'
img_savepath = '/home/tzheng2/workspace/scir-y1/demo_server/assets'
temp_video = 'output.mp4'
custom_js = """
<script>
    document.addEventListener('keydown', function(event) {
        if (event.key === ' ' && !event.shiftKey) {
            event.preventDefault();  // 防止空格键默认行为
            document.getElementById('submit-btn').click();  // 精确点击 submit_btn 按钮
        }
    });
</script>
"""

def encode_base64_str(imgfile):
    with open(imgfile, 'rb') as f:
        encoded_str = base64.b64encode(f.read())
    encoded_str = str(encoded_str)
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    return encoded_str

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)
 
def _launch_demo():
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'
                pic_idx += 1
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        pattern = r'<img>(.*?)</img>'
        match = re.search(pattern, message)
        if match:
            file_path = message[match.span()[0] + 5: match.span()[1] - 6]
            pre = 'C:\\Users\\tanthen\\AppData\\Local\\Temp\\gradio'
            file_path = '\\'.join(file_path.split('/')[-2:])
            file_path = os.path.join(pre, file_path)
            encoding_string = encode_base64_str(imgfile=file_path)
        else:
            encoding_string = ''
        data = {
            'message': message,
            'history': history,
            'img_base64': encoding_string
        }
        # response, history = model.chat(tokenizer, message, history=history)
        # responses = requests.post(url, files=files, data=data, stream=True)
        responses = requests.post(url, json=data, stream=True)
        # ipdb.set_trace()
        for response in responses.iter_content(chunk_size=1024):
            response = response.decode('utf-8')
            print(response)
            _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))
            yield _chatbot
            full_response = _parse_text(response)
        response = full_response
        history.append((message, response))
        _chatbot[-1] = (_parse_text(chat_query), response)
        task_history[-1] = (query, full_response)
        print("Qwen-VL-Chat: " + _parse_text(full_response))
        yield _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, message):
        for file_name in message["files"]:
            history = history + [((file_name,), None)]
            new_filename = file_name
            global img_savepath
            print(new_filename)
            # origin_path = new_filename[new_filename.span()[0] + 5: new_filename.span()[1] - 6]
            pattern = r'\\([^\\]+\\[^\\]+)$'
            match2 = re.search(pattern, new_filename)
            img_lastpath = new_filename[match2.span()[0]: match2.span()[1]][1:]
            img_saves = os.path.join(img_savepath, img_lastpath).replace('\\', '/')
            task_history = task_history + [((img_saves,), None)]
        if message["text"] is not None:
            text = message["text"]     
            task_text = text
            if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
                task_text = text[:-1]
            history = history + [(_parse_text(text), None)]
            task_history = task_history + [(task_text, None)]
        return history, task_history, gr.MultimodalTextbox(value=None, interactive=False)

    def reset_state(task_history):
        task_history.clear()
        return []

    def add_face(face_img):
        return gr.update(value=face_img, visible=True), gr.update(value = None, visible=False)

    def get_video(task_history, image):
        # ipdb.set_trace()
        text = task_history[-1][-1]
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, format="JPEG")
        
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data = {
            'text': text,
            'img_base64': image_base64
        }
        response = requests.post(url_image, json=data)
        with open(temp_video, 'wb') as f:
            f.write(response.content)
        return []
    def pre_video():
        return gr.update(visible=False), gr.update(value=temp_video, visible=True, autoplay=True)
    
    with gr.Blocks() as demo:
        gr.Markdown("""
<div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src='https://pic1.zhimg.com/70/v2-492772d7a1a0acc16f0e9a87c32ab9c8_1440w.avis?source=172ae18b&biz_tag=Post' style='width: 400px; height: auto; margin-right: 10px;' />
</div>
""")
        gr.Markdown("""
# SCIR-SC 多模态X具身智能 虚拟数字人
""")
        with gr.Row():
            with gr.Column(scale=1):
                video_upload_image = gr.Image(label='faceimage', height = 600, visible=True)
                video = gr.Video(label='video', height = 600, visible=False)
                addface_btn = gr.UploadButton("upload face (上传人脸)", file_types=['image'])
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label='Chat History', bubble_full_width=False, elem_classes="control-height", height=750)
        chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="输入文字", show_label=False)
        task_history = gr.State([])

        chat_input.submit(add_text, [chatbot, task_history, chat_input], [chatbot, task_history, chat_input]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True).then(
            lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]).then(
            get_video, [task_history, video_upload_image], []
            ).then(
                pre_video, [], [video_upload_image, video]
            )
        # with gr.Row():
        #     empty_bin = gr.Button("🧹 Clear History (清除历史)")
        #     regen_btn = gr.Button("🤔️ Regenerate (重试)")
        # empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        # regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addface_btn.upload(add_face, [addface_btn], [video_upload_image, video])
        
        
    demo.queue().launch(
        share=share,
        inbrowser=inbrowser,
        server_port=server_port,
        server_name=server_name
    )
    
def main():

    _launch_demo()


if __name__ == '__main__':
    main()
