import requests
import json
import cv2
import sys
import numpy as np
import cv2
import thread
import scipy.misc
import config as cfg
import time
import copy

w, h = 6, 100;
objects_mat = [[0 for x in range(w)] for y in range(h)]
num_objects = 0;
frame_rate = 0;

def call_YOLO(img):
    global frame_rate
    addr = cfg.SERVER_ADDRESS
    url = addr + '/api/edge_app2'

    # prepare headers for http request
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    # encode image as jpeg
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    #_, img_encoded = cv2.imencode('.jpg', img)

    start_time = time.time();
    # send http request with image and receive response
    try:
        response = requests.post(url, data=img_str, headers=headers)
        #response = requests.post(url, data=img_encoded.tostring(), headers=headers)
    except Exception:
        print "Connection problem!"
        return 0

    # decode response
    if response.text == "":
        return 0

    frame_rate = round(1./(time.time() - start_time),2);

    try:
        response_obj = json.loads(response.text);
    except Exception:
        return;

    #print (list(objects_list.keys()))
    #print (type(objects_list['results'][0]))

    return
    
    ##############
    i = 0;

    global objects_mat;
    global num_objects;
    num_objects = len(objects_list);

    while i < len(objects_list):
            d  = (objects_list[i])

            for key, value in d.items():
                    object_name = value[0];
                    object_accuracy = value[1];
                    object_pos = value[2];
                    pos = [0,0,0,0];
                    #print ("I'm %.2f%% sure I've seen a " + object_name + ".") %object_accuracy
                    for p, v in object_pos.items():
                            pos[0] = int(v[0]);
                            pos[1] = int(v[1]);
                            pos[2] = int(v[2]);
                            pos[3] = int(v[3]);

                    objects_mat[i][0] = pos[0]+pos[2];
                    objects_mat[i][1] = pos[1]+pos[3];
                    objects_mat[i][2] = pos[0]-pos[2];
                    objects_mat[i][3] = pos[1]-pos[3];
                    objects_mat[i][4] = object_name;
                    objects_mat[i][5] = object_accuracy;


            i = i + 1;
    return 0



### Video object detection app

print len(sys.argv)
if (len(sys.argv) == 1):
    cap = cv2.VideoCapture(0)
else:
    video_file = sys.argv[1];
    cap = cv2.VideoCapture(video_file)


num_frames = 0;
font = cv2.FONT_HERSHEY_DUPLEX

while(cap.isOpened()):

    # read video and obtain a frame
    ret, frame = cap.read()
    num_frames = num_frames + 1;

    # offloading logic
    if np.mod(num_frames,cfg.FRAME_RATE) == 0:
        thread.start_new_thread(call_YOLO,(copy.copy(frame),))

    # print objects rectangles on the frame
    i = 0;
    while i < num_objects:
        cv2.rectangle(frame,(objects_mat[i][0],objects_mat[i][1]),(objects_mat[i][2],objects_mat[i][3]),(0,255,0),1)
        label = objects_mat[i][4] + " " + str(round(objects_mat[i][5],2))
        #cv2.rectangle(imCrop,(objects_mat[i][2],objects_mat[i][3]-3),(objects_mat[i][2]+len(objects_mat[i][4])*10 + 40,objects_mat[i][3]-18),(0,0,0),-1)
        cv2.putText(frame, label, (objects_mat[i][2],objects_mat[i][3]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0))
        i = i + 1

    cv2.rectangle(frame,(10,15),(100,35),(0,0,0),-1)
    cv2.putText(frame,`frame_rate` + " fps",(20,30), font, 0.5,(0,255,0),1)

    # show the frame
    cv2.imshow('frame',frame)

    # option to quit the app (press 'q')
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
