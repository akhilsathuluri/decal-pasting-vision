import io
import json
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
# from flask import Flask, jsonify, request
import cv2
from node import node
from datetime import datetime

print('initialising model...')

model_name = 'resnet'

if model_name == 'resnet':
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    model.load_state_dict(torch.load("trained/new/tank_model_ident_res18_fe.pth"))
    model.eval()
    print('Resnet model initialised...')

if model_name == 'squeeze':
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))

    model.load_state_dict(torch.load("trained/new/tank_model_ident_squeeze_fe.pth"))
    model.eval()
    print('Squeezenet model initialised...')

# Not working yet
if model_name == 'mobile':
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, 4)

    model.load_state_dict(torch.load("trained/new/tank_models_mobile_fe.pth"))
    model.eval()
    print('Mobilenet_v2 model initialised...')

class_index = json.load(open('classes.json'))
print('classes loaded...')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (200,450)
fontScale              = 2
fontColor              = (255,255,0)
lineType               = 10
print('settings loaded...')

# template = cv2.imread('background.png')
# _, w, h = template.shape[::-1]
# method = eval('cv2.TM_CCOEFF')
# print('background template and method loaded...')

# def check_presence(frame):
#     res = cv2.matchTemplate(frame,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     if max_val < 320000000:
#         comp_present = 0
#     else:
#         comp_present = 1
#     return comp_present

def transform_image(img):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.fromarray(img)
    return my_transforms(image).unsqueeze(0)

sm = torch.nn.Softmax()
def get_prediction(tensor):
    outputs = model.forward(tensor)
    # _, y_hat = outputs.max(1)
    # predicted_idx = str(y_hat.item())
    prob = sm(outputs)
    print(prob)
    # return prob, class_index[predicted_idx]
    return prob

# def check_presence(frame):
#

# a = datetime.now()
# while True:
#     ret, frame = cam.read()

#     tensor = transform_image(frame)
#     # print("obtaining prediction...")
#     pred = get_prediction(tensor)
#     # print(pred)
#     # comp_present = check_presence(frame)

#     # if comp_present == 1:
#     #     pred = pred
#     # elif comp_present == 0:
#     #     pred = "No tank"
#         # pred = " "

#     cv2.putText(frame,pred,
#         bottomLeftCornerOfText,
#         font,
#         fontScale,
#         fontColor,
#         lineType)

#     cv2.imshow('Display', frame)
#     frames = frames + 1

#     k = cv2.waitKey(1)

#     if k%256 == 27:
#         print('Escape hit. Closing...')
#         break
#     elif k%256 == 32:
#         img_name = "image_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written".format(img_name))
#         img_counter += 1
# b = datetime.now()
# c = b-a
# cam.release()
# cv2.destroyAllWindows()
# print("framerate: ", np.float(frames)/np.float(c.seconds), "fps")

def create_node(node):
    # Instantiate node
    node = node.Node()
    # Set node params
    node.node_description = 'Decal pasting machine for petrol tanks of various models, with vision based model identification system'
    node.node_number = 'N1_1593'
    node.node_name = 'Decal pasting machine'

    #node.host_ip = '127.0.0.1'
    node.host_ip = '192.168.3.250'
    node.host_port = '502'

    # Initiate a database
    node.init_db(node.node_number)

    # Load register map
    map = node.load_register_map()
    # Connect with slave
    node.connect()
    # Reset the entire memory block under pi's control
    start_register = 50
    block_length = 15
    rq = node.client.write_registers(start_register, [0]*block_length, unit=node.unit)
    return node, map

if __name__=="__main__":
    # Setup camera feed
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Display")

    while True:        
        prob = np.zeros(4)
        for frames in range(3):
            prob += np.array(get_prediction(tensor))
            ret, frame = cam.read()
            tensor = transform_image(frame)
            prob = prob/3
            
        
        cv2.putText(frame,pred,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

        cv2.imshow('Display', frame)

        k = cv2.waitKey(1)

        if k%256 == 27:
            print('Escape hit. Closing...')
            break
        elif k%256 == 32:
            img_name = "image_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written".format(img_name))
            img_counter += 1
    
cam.release()
cv2.destroyAllWindows()