# python 2.7

import json
import sys
import time
import socket
import subprocess

ADB = "~/Library/Android/sdk/platform-tools/adb"
PHONE_SERVER_IP = "127.0.0.1"
PHONE_SERVER_PORT = 8080

def start_app():
    # make sure phone is awake and object tracker app is running on phone
    command = ADB + ' shell input keyevent KEYCODE_WAKEUP\n'
    command += ADB + ' shell am start -n "ie.tcd.netlab.objectracker/ie.tcd.netlab.objecttracker.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER'
    p = subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    print('%s%s'%(stdout,stderr))

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

def process_jpg(s,img_file):
    # send img_file (must be jpg) to phone and receive detection results back (as json)
    with open(img_file, mode='rb') as file:
        jpg = file.read()
    send_cmd(s,"JPG "+str(len(jpg)) + "\n", False)
    s.sendall(bytearray(jpg))
    read_line_ok(s, "JPG "+img_file)
    return read_line(s) # get the json response

start_app()
s=open_conn()
send_cmd(s, "SET use_camera false\n");
send_cmd(s, 'SET jpeg_quality 100\n')
json=process_jpg(s,'darknet/data/tennis.jpg'); print(json)
