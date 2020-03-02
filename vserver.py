from flask import Flask, request, Response
import json
import numpy as np
import cv2
import darknet as dn
import config as cfg
import time

# Initialize the Flask application
app = Flask(__name__)
net = dn.load_net(cfg.CONFIG_MODEL, cfg.MODEL_WEIGHTS, 0)
im_net = dn.load_net(cfg.IM_CONFIG_MODEL, cfg.IM_MODEL_WEIGHTS, 0)
dn.set_batch_network(net,1);
dn.resize_network(net,320,320);
meta = dn.load_meta(cfg.MODEL_DATA)
im_meta = dn.load_meta(cfg.IM_MODEL_DATA);

IM_CLASS = 1
users = dict();

def classify_image():
   
    im = dn.load_image('subframe.jpg',0,0);
    r = dn.classify(im_net,im_meta,im)
    
    return r[0]
    

def generate_response(yolo_objects):

    i = 0
    response_dict = dict();
    objects_list = [];

    while i < len(yolo_objects):
        d  = yolo_objects[i]
        object_dict = dict()
        object_dict['title'] = d[0]
        object_dict['confidence'] = d[1]
        i = i + 1
        pos = d[2]
        j = 1
        while j < len(pos):
            object_dict['x'] = pos[0];
            object_dict['y'] = pos[1];
            object_dict['w'] = pos[2];
            object_dict['h'] = pos[3];
            j = j + 1
        objects_list.append(object_dict)

    response_dict['results'] = objects_list
    timings = dict();
    timings['jpeg_encode'] = 0;
    timings['rot'] = 0;
    timings['net'] = 0;
    timings['yolo'] = 0;
    timings['jpeg_decode'] = 0;
    timings['server'] = 0;
    timings['size'] = 0;
    response_dict['server_timings'] = timings;

    return response_dict

def object_detection(request):

    # convert string of image data to uint8
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, 1)
    #frame = dn.nparray_to_image(img);
    cv2.imwrite('frame.jpg',img);
    #frame = dn.load_image('frame.jpg',0,0)
  
    r =  dn.detect(net, meta, 'frame.jpg')
    
    res = generate_response(r)
    
    if IM_CLASS  == 1:
        t = res['results']

        if len(t) > 0 :
            s = t[0]
            x = int(s['x'])
            y = int(s['y'])
            w = int(s['w'])
            h = int(s['h'])

            crop_img = img[max(int(y-h/2),0):max(int(y-h/2),0)+h, max(int(x-w/2),0):max(int(x-w/2),0)+w]
            cv2.imwrite('subframe.jpg',crop_img);
            t = classify_image()
            s['title'] = t[0];
            s['confidence'] = t[1];

    return res

# this is the app that works with the desktop app
@app.route('/api/edge_app2', methods=['POST'])
def edge_app():

    global users
    user_addr = request.remote_addr;
    if user_addr in users.keys():
        #response_pickled = jsonpickle.encode([])
        return Response(status=200)
    else:
        users[user_addr] = 1
        yolo_objects = object_detection(request)
        #response_object = generate_response(yolo_objects)
        #print yolo_objects
        
        #r = response_object['results'];
         

        
        jason_response = json.dumps(yolo_objects);
        users.pop(user_addr)

        # send response to the client
        return Response(response=jason_response, status=200, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=cfg.PORT, threaded=True)
