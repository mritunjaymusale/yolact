

import cv2  # Optional, see below
import ffmpeg
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from torch.nn import CosineSimilarity
from torchvision import transforms

from data import set_cfg
from layers.output_utils import postprocess
from utils.augmentations import FastBaseTransform
from yolact import setupYolact
from backend_setup import initial_setup
import video_utils

initial_setup()







net = setupYolact()


net = setupYolact()

transform = FastBaseTransform()
input_video_name = 'out.mp4'
width, height, fps = video_utils.getVideoMetadata(input_video_name)

process1 = video_utils.readVideo(input_video_name)




in_bytes = process1.stdout.read(width * height * 3)

in_frame = (
    np
    .frombuffer(in_bytes, np.uint8)
    .reshape([height, width, 3])
)

# actual NN part
tensor_image = torch.from_numpy(in_frame).cuda().float()
tensor_image_4d = tensor_image[None, ...]
tensor_image_4d = transform(tensor_image_4d)
preds = net(tensor_image_4d)
classes, scores, boxes, masks = postprocess(
    preds, width, height, score_threshold=0.25)

temp_boxes = boxes.detach().cpu().numpy()
for index,box in enumerate(temp_boxes):
    color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
    frame = cv2.putText(in_frame,str(index),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1,color,2)
    cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,2)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imwrite('first_frame.png',frame)

    
    

