from node import node
import streamlit as st
from datetime import datetime
from cycle import *
# from streamlit.ReportThread import add_report_ctx
import pandas as pd
from PIL import Image
from streamlit.ReportThread import add_report_ctx
import SessionState

import cv2
import os
import signal
import threading

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

node, map = create_node(node)
node.init_page()

if __name__=="__main__":
    ss = SessionState.get(image_counter=0)
    # Setup camera feed
    cam = cv2.VideoCapture(0)

    # image_counter = 0
    feed = st.empty()
    # save_button = st.button("Save frame")
    # cam.release()

    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        feed.image(frame, caption = 'Live feed from camera', use_column_width=True)
        cam.release()

        # if save_button:
        #     img_name = "./data/CT100/st_image_{}.png".format(ss.img_counter)
        #     cv2.imwrite(img_name, frame)
        #     st.write("{} written".format(img_name))
        #     ss.image_counter = ss.image_counter + 1
        #     cam.release()
    
    cam.release()