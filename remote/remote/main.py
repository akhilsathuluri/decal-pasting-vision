from node import node
import streamlit as st
from datetime import datetime
from cycle import *
# from streamlit.ReportThread import add_report_ctx
import pandas as pd
from PIL import Image

def create_node(node):
    # Instantiate node
    node = node.Node()
    # Set node params
    node.node_description = 'Decal pasting machine for petrol tanks of various models, with vision based model identification system'
    node.node_number = 'N1_1593'
    node.node_name = 'Decal pasting machine'

    node.host_ip = '127.0.0.1'
    #node.host_ip = '192.168.0.109'
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
    node, map = create_node(node)
    # Start the page
    node.init_page()
    st.write(node.engine)
    st.write(pd.DataFrame(map.items()))

    image = Image.open('tank.png')
    st.image(image, caption='Last identified model: '+'CT100', use_column_width=True)

    # Thread-1: heartbeat
    # Thread-2: cycle (model_identification)
    # Thread-3: database writing (optional)
    # Thread-4: streamlit show (can stream images?)
