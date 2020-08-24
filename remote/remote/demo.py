import io
import json

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
# from flask import Flask, jsonify, request
import cv2

# plt.ion()
print('initialising model...')
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model.load_state_dict(torch.load("trained/new/tank_model_ident_res18_fe.pth"))
model.eval()
print('model initialised...')

class_index = json.load(open('classes.json'))
print('classes loaded...')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (200,450)
fontScale              = 2
fontColor              = (0,0,0)
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

def get_prediction(tensor):
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    # print(predicted_idx)
    return class_index[predicted_idx]

# def check_presence(frame):
#

cam = cv2.VideoCapture(0)
cv2.namedWindow("Display")

img_counter = 0
comp_present = 0

while True:
    ret, frame = cam.read()

    tensor = transform_image(frame)
    # print("obtaining prediction...")
    pred = get_prediction(tensor)
    # print(pred)
    # comp_present = check_presence(frame)

    # if comp_present == 1:
    #     pred = pred
    # elif comp_present == 0:
    #     pred = "No tank"
        # pred = " "

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
