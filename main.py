import warnings
warnings.filterwarnings("ignore")

import ffmpeg
import numpy as np
import torch
from torch.nn import CosineSimilarity
from layers.output_utils import postprocess
from utils.augmentations import FastBaseTransform
from yolact import setupYolact
from backend_setup import initial_setup
import cv2
import video_utils 
import argparse
from sort import *




initial_setup()


net = setupYolact()


parser = argparse.ArgumentParser()
parser.add_argument("input_video_name",type=str)
parser.add_argument("output_video_name",type=str)
args = parser.parse_args()

transform = FastBaseTransform()
input_video_name =args.input_video_name
output_video_name =args.output_video_name
sort = Sort()




input_video_width, input_video_height, input_video_fps = video_utils.getVideoMetadata(input_video_name)

def doImageSegmentation(input_image):
    tensor_image = torch.from_numpy(input_image).cuda().float()
    tensor_image_4d = tensor_image[None, ...]
    tensor_image_4d = transform(tensor_image_4d)
    preds = net(tensor_image_4d)
    classes, scores, boxes, masks = postprocess(
        preds, input_video_width, input_video_height, score_threshold=0.15)
    return boxes, tensor_image, masks,scores

cap = cv2.VideoCapture(input_video_name)
ret,frame = cap.read()
if ret:
    boxes, tensor_image, masks, scores = doImageSegmentation(frame)
    trackers =sort.update(boxes.int().detach().cpu().numpy())
    for index,box in enumerate(trackers):
        box = np.array(box,np.int32)
        color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
        frame = cv2.putText(frame,str(box[4]),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,1,color,2)
        cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,2)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite('first_frame.png',frame)
else:
    print("Something went wrong with OpenCV first frame input, quitting...")
    quit()


# cant take commandline inputs after the ffmpeg prompt happens
mask_id = int(input('Enter the mask id from first_frame.png:'))




videoToFrames = video_utils.readVideo(input_video_name)

framesToVideo = video_utils.writeVideo(output_video_name,input_video_width,input_video_height,input_video_fps)







def returnTensorIndexWithClosestSimilarity(previous_box,boxes):
    '''
    Returns the index location of array of bounding boxes
    '''
    previous_box = previous_box.repeat(int(boxes.shape[0]),1) 
    simalirity = cosine_sim(boxes.float(),previous_box.float())
    best_similarity = torch.max(simalirity)
    mask_index_location = ((simalirity==best_similarity).nonzero())
    return mask_index_location

def getCustomBB(tensor_image,cv2_window_name):
    temp_image = np.array(tensor_image.int().detach().cpu().numpy(),np.uint8)
    customBB = cv2.selectROI(cv2_window_name,cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB ),fromCenter=False,
			showCrosshair=True)
    return customBB
 



##### Now to use ######
count = 0
previous_box=None
cosine_sim= CosineSimilarity()

RGB_CHANNELS=3

# Useful when the target is out of sight for the rest of the video, just send it for encoding phase rather going through the network and frame manipulation 
PASSTHROUGH_MODE = False

cv2_window_name ="output"
cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)    

while True:
    in_bytes = videoToFrames.stdout.read(input_video_height * input_video_width * RGB_CHANNELS)
    if not in_bytes:
        break
    in_frame = (
        np
        .frombuffer(in_bytes, np.uint8)
        .reshape([input_video_height, input_video_width, RGB_CHANNELS])
    )
    if PASSTHROUGH_MODE:
        out_frame = in_frame
    else:
        
        # actual NN part
        boxes, tensor_image, masks, scores = doImageSegmentation(in_frame)

        


        trackers = sort.update(boxes.int().detach().cpu().numpy())
        
        #check if humans in frame
        if boxes.shape!=torch.Size([0]):
        
        
        
            # human detected in frame
            # find the box based user's mask id 
            required_bounding_box = trackers[np.where(trackers[:,4]==mask_id)]
            if required_bounding_box.size == 0:

                print("Target not in sight, do custom bounding boxes")
                customBB= getCustomBB(tensor_image,cv2_window_name)
                if customBB == (0,0,0,0):
                    print("Entering passthrough mode, you won't be able to interact with frames now")
                    cv2.destroyWindow(cv2_window_name)
                    PASSTHROUGH_MODE=True
                customBB = torch.Tensor(customBB).cuda().int()
                tensor_image[customBB[1]:customBB[3],customBB[0]:customBB[2],: ] = 0

            else :
                required_bounding_box =required_bounding_box[0][0:4]
                required_bounding_box = torch.Tensor(required_bounding_box).cuda().int()

                mask_index_location = returnTensorIndexWithClosestSimilarity(required_bounding_box,boxes)
                

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
print("Finished processing video")
cv2.destroyAllWindows()
# Merge the audio of the source video with the created video 

print("Merging audio from source video")
source_audio = video_utils.getAudio(input_video_name)
new_video = video_utils.getVideo(output_video_name)
# fine tune the code settings
final_video = ffmpeg.output(new_video,source_audio,'audio_merged_'+output_video_name, vcodec='copy',acodec='copy')

final_video.run(overwrite_output=True,quiet=True)

quit()

# You can modify the score threshold to your liking



