import streamlit as st
from datetime import datetime
from PIL import Image

from node import node

node = node.Node()
# Set node params
node.node_description = 'Decal pasting machine for petrol tanks of various models, with vision based model identification system'
node.node_number = 'N1_1593'
node.node_name = 'Decal pasting machine'

node.init_page()

recent_frame = st.empty()

image = Image.open('database/recent_tank.jpg')
recent_frame.image(image, caption='Last identified tank', use_column_width=True)
