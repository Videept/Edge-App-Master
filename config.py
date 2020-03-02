# Server parameters

DARKNET_PATH = "darknet/"
CONFIG_MODEL = DARKNET_PATH + "cfg/yolov3.cfg"
MODEL_WEIGHTS = DARKNET_PATH + "yolov3.weights"
MODEL_DATA = DARKNET_PATH + "cfg/coco.data"

IM_CONFIG_MODEL = DARKNET_PATH + "cfg/densenet201.cfg"
IM_MODEL_WEIGHTS = DARKNET_PATH + "densenet201.weights"
IM_MODEL_DATA = DARKNET_PATH + "cfg/imagenet1k.data"

# Client parameters

PORT = 8000    
SERVER_ADDRESS = "http://pine.scss.tcd.ie:" + `PORT`
#SERVER_ADDRESS = "http://localhost:" + `PORT`
#SERVER_ADDRESS = "http://lily.scss.tcd.ie:" + `PORT`
#SERVER_ADDRESS = "http://192.168.1.253:" + `PORT`


FRAME_RATE = 1; # e.g. if 10, YOLO only process 1 of every 10 frames
TMP_FOLDER = "tmp/"
