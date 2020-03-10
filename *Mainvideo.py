import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import utils
import model as modellib
import visualize
import recombine as re
import video
import layout
import correctimages as ci

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_video.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(video.VideoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG','person','screen']

################## Customization ####################
input_name = 'CV.mp4'
output_name = 'CVleft.mp4'    # The finally video name format likes xxx_f.mp4
ROOT_DIR = os.getcwd()
input_path = os.path.join(ROOT_DIR,input_name)
output_path = os.path.join(ROOT_DIR,"videooutput/",output_name)

###########################################

capture = cv2.VideoCapture(input_path)
total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = int(round(capture.get(cv2.CAP_PROP_FPS),0))
total_frame = total_frame
print("The total frames are {}, frames per second are {}.".format(total_frame,FPS))

# Detect the first frame
ret, frame = capture.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = model.detect([frame], verbose=0)
r = results[0]
visualize.display_instances(frame,r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
#print(r['class_ids'],r['masks'])
dic = dict(zip(r['class_ids'],r['rois']))
person, screen = dic[1], dic[2]
person_y, person_x , person_h , person_w = person[0],person[1],person[2],person[3]
screen_y, screen_x, screen_h, screen_w  = screen[0]-10,screen[1],screen[2],screen[3]+50
#print(screen_y, screen_x, screen_h, screen_w)
#print(person_y, person_x , person_h , person_w)
print('Detection is completed!')


################### correct image and set background ##################
screen_crop = frame[screen_y:screen_h,screen_x:screen_w]
rect = ci.orderPoints(screen_crop)
warped =ci.fourPointsTransform(screen_crop,rect)

plt.imshow(warped)
plt.show()

NL = layout.NewLayout(warped, 10, person = "Yes", pp = "left")
bg, screenroi, personroi = NL.setbg()

videoWrite = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),FPS,(bg.shape[1],bg.shape[0]))
print('Generate the background completed')

################# Creat the tracking instance ###################
tracker = cv2.MultiTracker_create()
init_once = False
#tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = cv2.TrackerMOSSE_create()
# init the bbox size#######
bbox_p= (person_x, person_y, person_w-person_x, person_h-person_y)  # (x,y,w,h)   crop(y:h,x:w)
bbox_s = (screen_x,screen_y,screen_w-screen_x,screen_h-screen_y)

start_time = time.time()
for i in range(0,total_frame):
    ret, frame = capture.read()

    ############################ loading tracker ##############################
    if not init_once:
        ret = tracker.add(cv2.TrackerMOSSE_create(), frame, bbox_p)
        ret = tracker.add(cv2.TrackerMOSSE_create(), frame, bbox_s)
        init_once = True

    ret, boxes = tracker.update(frame)
    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))

    person_bbox = list(map(int,boxes[0]))
    screen_bbox = list(map(int,boxes[1]))
    #print('Tracking frame {}, person bbox is {}, screen bbox is {}.'.format(i, person_bbox, screen_bbox))

    try:

        #s_bbox_x, s_bbox_y, s_bbox_w, s_bbox_h = screen_bbox[0], screen_bbox[1], screen_bbox[2], screen_bbox[3]
        #screen_crop = frame[s_bbox_y:s_bbox_y + s_bbox_w, s_bbox_x:s_bbox_x + s_bbox_w]
        #rect = ci.orderPoints(screen_crop)

        screen_crop = frame[screen_y:screen_h,screen_x:screen_w]
        warped = ci.fourPointsTransform(screen_crop, rect)
        bg = NL.format(frame, bg, person_bbox, warped, screenroi, personroi)

        videoWrite.write(bg)
        print('Convert the frame {} to a video'.format(i))

    except TypeError:
        pass

end_time = time.time()
interval = end_time - start_time
capture.release()
videoWrite.release()
############################ Combine the audio and video ##################
re.combine_both(output_path,input_path)
##########################################################
print('The interval is {}s '.format(interval))
