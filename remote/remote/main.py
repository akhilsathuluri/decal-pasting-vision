from node import node
import streamlit as st
from datetime import datetime
from cycle import *
# from streamlit.ReportThread import add_report_ctx
import pandas as pd
from PIL import Image
from streamlit.ReportThread import add_report_ctx
import SessionState
from model_identification import check_model

import cv2
import time
import os
import signal
import threading
import multiprocessing as mp

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
    rmap = node.load_register_map()
    # Connect with slave
    node.connect()
    # Reset the entire memory block under pi's control
    start_register = 50
    block_length = 15
    rq = node.client.write_registers(start_register, [0]*block_length, unit=node.unit)
    return node, rmap

node, rmap = create_node(node)
# node.init_page()

# recent_image = st.empty()
# status = st.empty()

if __name__=="__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Prediction display')

    p1 = mp.Process(target = health, args=(node, rmap, ))
    # p2 = mp.Process(target = cycle, args=(node, rmap, ))

    p1.start()

    prev_model = 'None'
    while True:
        ret, frame = cam.read()
        # pred, frame = check_model(frame)

        trigger1 = node.client.read_holding_registers(rmap['reg_plc_trigger1'], 1, unit=node.unit)
        # temp = trigger1.registers[0]
        # if trigger1.registers[0] == 1:
        if True:
            pred, frame = check_model(frame)
            if pred != 'None':
                # print(pred)
                # recent_image.image(frame, caption=pred, use_column_width=True)
                # status.write('Component present')
                if pred != prev_model:
                    # Write model verify register
                    rq = node.client.write_registers(rmap['reg_pi_{}'.format(pred)], 1, unit=node.unit)
                    cv2.imwrite('database/recent_tank.jpg', frame)
                    prev_model = pred
                else:
                    # status.write('Component absent')
                    pass

        cv2.imshow('Prediction display', frame)
        print(pred)

        k = cv2.waitKey(1)
        if k%256== 27:
            print('Escape hit. Closing...')
            break

    cam.release()
    cv2.destroyAllWindows()

    # p2.start()
    p1.join()
    # p2.join()