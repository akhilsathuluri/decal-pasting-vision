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
from datetime import datetime

print('initialising model...')

model_name = 'resnet'

if model_name == 'resnet':
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)

    model.load_state_dict(torch.load("trained/new/tank_models_hole.pth"))
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
fontColor              = (255,0,0)
lineType               = 10
print('settings loaded...')

template = cv2.imread('background.png')
_, w, h = template.shape[::-1]
method = eval('cv2.TM_CCOEFF')
print('background template and method loaded...')

# Component precense
def check_presence(frame):
    res = cv2.matchTemplate(frame,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)
    if max_val > 11454307000:
        comp_present = 0
    else:
        comp_present = 1
    return comp_present

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
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    prob = sm(outputs)
    print(prob)
    # if torch.max() > 50:
    #     print(yo)
    return class_index[predicted_idx], torch.max(prob).detach().numpy()

cam = cv2.VideoCapture(0)
cv2.namedWindow("Display")

img_counter = 0
comp_present = 0
frames = 1
a = datetime.now()
while True:
    ret, frame = cam.read()
    tensor = transform_image(frame)
    # print("obtaining prediction...")
    pred, prob = get_prediction(tensor)
    status = check_presence(frame)
    if status==1:
        if prob > 0.45:
            # cv2.putText(frame,pred+':{}'.format(prob)+'%',
            #     bottomLeftCornerOfText,
            #     font,
            #     fontScale,
            #     fontColor,
            #     lineType)
            cv2.putText(frame,pred,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
        else:
            cv2.putText(frame,"Please wait",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    if status==0:
        cv2.putText(frame,"Waiting for tank",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('Display', frame)
    frames = frames + 1

    k = cv2.waitKey(1)

    if k%256 == 27:
        print('Escape hit. Closing...')
        break
    elif k%256 == 32:
        img_name = "image_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written".format(img_name))
        img_counter += 1

b = datetime.now()
c = b-a
cam.release()
cv2.destroyAllWindows()
print("framerate: ", np.float(frames)/np.float(c.seconds), "fps")
