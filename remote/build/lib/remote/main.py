from node import node
import streamlit as st
from datetime import datetime
# from streamlit.ReportThread import add_report_ctx
import pandas as pd

def create_node(node):
    # Instantiate node
    node = node.Node()
    # Set node params
    # node.node_description = 'Pad printing machine for Husqvarna rims'
    # node.node_number = 'N1_1507'
    # node.node_name = 'Pad printing machine'

    node.host_ip = '127.0.0.1'
    #node.host_ip = '192.168.0.109'
    node.host_port = '502'

    # Initiate a database
    node.init_db()

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
    st.write(datetime.now())
