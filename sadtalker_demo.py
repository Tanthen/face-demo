from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
import uvicorn
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import base64
from pydantic import BaseModel
import requests

url = 'http://127.0.0.1:9999/fish'

def decode_base64_str(encoded_str, save_path):
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    decoding_bytes = base64.b64decode(encoded_str)
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(save_path, 'wb') as f:
        f.write(decoding_bytes)

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


result_dir = './results'
pose_style=0
batch_size=2
size=256
expression_scale=1
input_yaw = None
input_pitch = None
input_rollt = None
enhancer='gfpgan'
background_enhancer = None
cpu = False
face3dvis=False
still = False
preprocess='crop'
verbose = False
old_version = False
net_recon='resnet50'
use_last_fc = False
bfm_folder = './checkpoints/BFM_Fitting/'
bfm_model = 'BFM_model_front.mat'
focal=1015
center=112
camera_d=10
z_near=5
z_far=15


if torch.cuda.is_available() and not cpu:
    device = "cuda"
else:
    device = "cpu"


save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
os.makedirs(save_dir, exist_ok=True)
pose_style = pose_style
device = device
batch_size = batch_size
input_yaw_list = input_yaw
input_pitch_list = input_pitch
input_roll_list = None
ref_eyeblink = None
ref_pose = None
checkpoint_dir = './checkpoints'
current_root_path = ''
sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

#init model
preprocess_model = CropAndExtract(sadtalker_paths, device)

audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)

animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

def main(audio_path, pic_path):
    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)
    
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    
    shutil.move(result, save_dir+'.mp4')
    
    print('The generated video is named:', save_dir+'.mp4')

    if not verbose:
        shutil.rmtree(save_dir)

    return save_dir + '.mp4'



class Item(BaseModel):
    text: str
    img_base64: str

face_temp_path = "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/Wav2Lip/temp/face_temp.png"
audio_temp_path = '/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/fish-speech/fish-speech-main/fake.wav'

@app.post("/getv")
def videogen(item: Item):
    global face_temp_path
    global audio_temp_path
    decode_base64_str(item.img_base64, face_temp_path)
    data = {
        "text": item.text,
        "role": "Klee"
    }
    requests.post(url, json=data)
    
    gen_path = main(audio_temp_path, face_temp_path)
    print('=' * 100)
    print(gen_path)
    return FileResponse(gen_path, media_type='video/mp4')
