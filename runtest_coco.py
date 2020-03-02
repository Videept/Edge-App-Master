# python 2.7

import json
import sys
from tqdm import tqdm
import time
import socket
import subprocess
import cv2
import os
import statistics
from pycocotools.cocoeval import COCOeval # need to install coco eval tools
from pycocotools.coco import COCO

ADB = "adb"
PHONE_SERVER_IP = "127.0.0.1" #""192.168.1.48"
PHONE_SERVER_PORT = 8080
COCO_DIR = "coco"
IMG_STUB = COCO_DIR+"/images/val2014/" # add image_id+".jpg" to get filename
ANNOTATION_STUB = COCO_DIR + "/annotations/"
ANNOTATIONS = COCO_DIR + "/annotations/instances_val2014.json"

def start_app():
    command = ADB + ' shell input keyevent 26' #KEYCODE_WAKEUP\n' # wake up phone
    command += ADB + " forward tcp:8080 tcp:8080\n" # forward localhost:8080 to phone:8080
    command += ADB + ' shell am start -n "ie.tcd.netlab.objectracker/ie.tcd.netlab.objecttracker.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER'
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print('%s%s'%(stdout,stderr))
    time.sleep(1) # might take a while for phone to wake up

def open_conn():
    # open connection to server on phone
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((PHONE_SERVER_IP, PHONE_SERVER_PORT))
    return s

def read_line(s):
    line=""
    while 1:
         res = s.recv(1)
         if (not res) or (res == '\n'): break
         line += res.decode("ascii")
    return line

def read_line_ok(s, cmd):
    # if response is not 'ok', bail
    resp = read_line(s)
    if (resp != 'ok'):  # something went wrong
        print(cmd + " : " + resp)
        exit(-1)

def send_cmd(s,cmd, readline = True):
    # send a command to the phone
    msg = bytearray()
    msg.extend(cmd.encode("ascii"))
    s.sendall(msg)
    if (readline):
        read_line_ok(s,cmd)

def process_jpg(s,jpg):
    # send img (must be jpg) to phone and receive detection results back (as json)
    #with open(img_file, mode='rb') as file:
    #    jpg = file.read()
    send_cmd(s,"JPG "+str(len(jpg)) + "\n", False)
    s.sendall(bytearray(jpg))
    read_line_ok(s, "JPG "+img_file)
    return read_line(s) # get the json response

def get_category_id(cat, cats):
    # map from yolo to coco labels
    if cat=='tvmonitor':
        cat = 'tv'
    if cat == 'diningtable':
        cat = 'dining table'
    if cat == 'pottedplant':
        cat = 'potted plant'
    if cat == 'sofa':
        cat = 'couch'
    if cat == 'aeroplane':
        cat = 'airplane'
    if cat == 'motorbike':
        cat = 'motorcycle'
    try:
        res=(item for item in cats if item["name"] == cat).next()
        return res['id']
    except:
        return -1

def get_id_category(id):
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    try:
        res=(item for item in cats if item["id"] == id).next()
        return res['name']
    except:
        return -1

def show_ground_truth(i, img=None, verbose=1):
     truth = cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=i))
     for t in truth:
         if verbose:
             print(t["image_id"], t["category_id"], t["bbox"])
         x = int(t["bbox"][0]); y = int(t["bbox"][1]); w = int(t["bbox"][2]); h = int(t["bbox"][3]);
         if img.any(): # overlay ground truth on image (marked in red)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(img, get_id_category(t["category_id"]), (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

#images=[42, 74, 133, 136, 139, 143, 164, 192, 196, 208]
images = os.listdir(IMG_STUB);
cocoGt = COCO(ANNOTATIONS)
cats = cocoGt.loadCats(cocoGt.getCatIds())

start_app()
s=open_conn()
send_cmd(s, "SET use_camera false\n")
send_cmd(s, 'SET jpeg_quality ' + sys.argv[2] + '\n')
if (sys.argv[1] == 'udp'):
    send_cmd(s, "SET udp true\n")
else:
    send_cmd(s, "SET udp false\n")

dds = []
imgIds = []

#cv2.namedWindow('image'); cv2.moveWindow('image',0,0) # move window to side of screen

timings={'size':0, 'jpeg_decode':0, 'rot': 0, 'jpeg_encode':0, 'yolo': 0, 'net':0, 'client':0, 'server':0}
timings_total={'size':[], 'jpeg_decode':[], 'rot': [], 'jpeg_encode':[], 'yolo': [], 'net':[], 'client':[], 'server':[]}
timings_var={'size':[], 'jpeg_decode':[], 'rot': [], 'jpeg_encode':[], 'yolo': [], 'net':[], 'client':[], 'server':[]}

count=0;
image_dropped_count = 0;
iters = 5000; #len(images); #50;
resultsall_client=[]; resultsall_server=[]

#send_cmd(s,"JPGS "+str(iters)+"\n", False);
resall=[]
for j in tqdm(range(0,iters)):
    i = images[j];
    img_file = IMG_STUB+i;
    with open(img_file, mode='rb') as file:
        jpg = file.read()
    # call yolo via phone
    send_cmd(s,"JPG "+str(len(jpg)) + "\n", False)
    s.sendall(bytearray(jpg))
    read_line_ok(s, "JPG "+str(i))
    resall.append(read_line(s))

for j in tqdm(range(0,iters)):
#for j in tqdm(range(len(images))):
    i = images[j];
    id  = i.replace('COCO_val2014_','');
    id  = int(float(id.replace('.jpg','')))
    res=resall[j];
    detects = [];
    try:
        detects = json.loads(res)
    except ValueError:  # includes simplejson.decoder.JSONDecodeError
        image_dropped_count+=1;
    if detects == [] or detects['server_timings'].keys() == []:
	      print('Image dropped')
        #print(detects)
	      image_dropped_count+=1;
    else:
        count+=1
        if (count>1): # ignore first image as slower due to wifi wakeup
            imgIds.append(id)
            resultsall_client.append(detects['client_timings']);
            resultsall_server.append(detects['server_timings']);
        print(detects['server_timings']['size'],detects['client_timings']['yuvtoJPG'],detects['client_timings']['url2'],detects['client_timings']['url3']);
            timings['jpeg_decode']+=detects['server_timings']['jpg']
	          timings_total['jpeg_decode'].append(detects['server_timings']['jpg'])
            timings['rot']+=detects['server_timings']['rot']
	          timings_total['rot'].append(detects['server_timings']['rot'])
            timings['yolo']+=detects['server_timings']['yolo']
	          timings_total['yolo'].append(detects['server_timings']['yolo'])
            timings['server']+=detects['server_timings']['tot']
	          timings_total['server'].append(detects['server_timings']['tot'])
            timings['jpeg_encode']+=detects['client_timings']['yuvtoJPG']
	          timings_total['jpeg_encode'].append(detects['client_timings']['yuvtoJPG'])
            timings['client']+=detects['client_timings']['d']
	          timings_total['client'].append(detects['client_timings']['d'])
            timings['net'] += detects['client_timings']['url2']+detects['client_timings']['url3'] - detects['server_timings']['tot']
                          # -detects['server_timings']['yolo']-detects['server_timings']['jpg']\
                          # -detects['server_timings']['rot']
	          timings_total['net'].append(detects['client_timings']['url2']+detects['client_timings']['url3'] - detects['server_timings']['tot'])
            timings['size']+=detects['server_timings']['size']
	          timings_total['size'].append(detects['server_timings']['size'])
            # convert output to coco results format
            for result in detects['results']:
                r=result['right']; l=result['left']; b=result['bottom']; t=result['top']
                x=int(l); y=int(t); w=int(r-l); h=int(b-t)
                cat_id = get_category_id(result['title'],cats)
                if cat_id<0:
                    print("category "+result['title']+ " not found")
                    continue
                dds.append({'image_id':id, 'category_id':cat_id, 'bbox':[x,y,w,h],
                        'score':result['confidence'],
                        'client_timings':detects['client_timings'],
                        'server_timings':detects['server_timings']})
#read_line_ok(s,"JPGS");
#print(timings)
s.close()

with open('results_client','w') as f:
    f.write(json.dumps(resultsall_client))
with open('results_server','w') as f:
    f.write(json.dumps(resultsall_server))

# save the detection results in a file for later
with open('results.yolo', 'w') as f:
    f.write(json.dumps(dds))

cocoDt = cocoGt.loadRes(dds)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# output average timing measurements
for k in timings.keys():
    timings[k] = timings[k]/count
print(timings)

# Calculate variance of measurements
for key,val in timings_total.items():
    timings_var[key].append(timings[key] - (statistics.stdev(timings_total[key]))/2)
    timings_var[key].append(timings[key] + (statistics.stdev(timings_total[key]))/2)
print(timings_var)

setup_name = sys.argv[1] + '_' + 'JPG' + sys.argv[2] + '_YOLO' + sys.argv[3] + "\t";
with open('summary.txt', 'a+') as f:
    f.write(setup_name);
    f.write(json.dumps(timings))
    f.write("\t")
    f.write(str(image_dropped_count))
    f.write("\n")
    f.write(json.dumps(timings_var))
    f.write("\n\n")
f.close();
