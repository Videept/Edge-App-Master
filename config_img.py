# Server parameters

DARKNET_PATH = "/Users/victor/Applications/darknet/"
CONFIG_MODEL = DARKNET_PATH + "cfg/yolov3-tiny.cfg"
MODEL_WEIGHTS = DARKNET_PATH + "yolov3-tiny.weights"
MODEL_DATA = DARKNET_PATH + "cfg/coco.data"

# Client parameters

PORT = 8000 
#SERVER_ADDRESS = "http://192.168.6.131:" + `PORT`
#SERVER_ADDRESS = "http://localhost:" + `PORT`
#SERVER_ADDRESS = "http://oak.scss.tcd.ie:" + str(PORT)
#SERVER_ADDRESS = "http://lily.scss.tcd.ie:" + str(PORT)
SERVER_ADDRESS = "http://beech.scss.tcd.ie:" + str(PORT)


FRAME_RATE = 1; # e.g. if 10, YOLO only process 1 of every 10 frames
TMP_FOLDER = "tmp/"
