import ffmpeg
import numpy as np
import torch
from torch.nn import CosineSimilarity

from layers.output_utils import postprocess
from utils.augmentations import FastBaseTransform
from yolact import setupYolact
from backend_setup import initial_setup

import video_utils 

initial_setup()


net = setupYolact()

transform = FastBaseTransform()
input_video_name = 'tucker.mp4'
output_video_name = 'final.mp4'
input_video_width, input_video_height, input_video_fps = video_utils.getVideoMetadata(input_video_name)

# cant take commandline inputs after the ffmpeg prompt happens
mask_id = int(input('Enter the mask id from first_frame.png:'))

videoToFrames = video_utils.readVideo(input_video_name)

framesToVideo = video_utils.writeVideo(output_video_name,input_video_width,input_video_height,input_video_fps)


##### Now to use ######
count = 0
previous_box=torch.zeros([1,4]).cuda().float()
cosine_sim= CosineSimilarity()


RGB_CHANNELS=3

while True:
    in_bytes = videoToFrames.stdout.read(input_video_width * input_video_height * RGB_CHANNELS)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([input_video_height, input_video_width, RGB_CHANNELS])
    )

    # actual NN part
    tensor_image = torch.from_numpy(in_frame).cuda().float()
    tensor_image_4d = tensor_image[None, ...]
    tensor_image_4d = transform(tensor_image_4d)
    preds = net(tensor_image_4d)
    classes, scores, boxes, masks = postprocess(
        preds, input_video_width, input_video_height, score_threshold=0.25)
    
    # preserving the bounding boxes from previous frame
    previous_box= boxes[mask_id]
    previous_box = previous_box.repeat(int(boxes.shape[0]),1) 
    simalirity = cosine_sim(boxes.float(),previous_box.float())
    best_similarity = torch.max(simalirity)
    mask_index_location = ((simalirity==best_similarity).nonzero())
    
    previous_box = boxes[mask_index_location[0].item()]

    # actual frame manipulation 
    tensor_image[masks[mask_index_location[0].item()] == 1] = 0
    out_frame = tensor_image.int().detach().cpu().numpy()

    framesToVideo.stdin.write(
        out_frame
        .astype(np.uint8)
        .tobytes()
    )



framesToVideo.stdin.close()
videoToFrames.wait()
framesToVideo.wait()


source_audio = video_utils.getAudio(input_video_name)
new_video = video_utils.getVideo(output_video_name)
final_video = ffmpeg.output(new_video,source_audio,'test.mp4')
final_video.run()

quit()

# You can modify the score threshold to your liking



