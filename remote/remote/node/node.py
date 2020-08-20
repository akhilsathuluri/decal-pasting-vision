import streamlit as st
import json
from datetime import datetime
import time
from sqlalchemy import create_engine

from comm import comm

class Node(comm.Comm):
    def __init__(self):
        self.node_description = 'Insert node description by accessing node_description'
        self.node_number = '000'
        self.node_name = 'Insert node name by accessing node_name'
        comm.Comm.__init__(self)
        self.data = {}
        self.logging_status = False
        self.time_stamp = datetime.now()
        # Time in seconds
        self.log_frequency = 1

    def init_page(self):
        st.title("{}: {}".format(self.node_number, self.node_name))
        st.header(self.node_description)
        st.text('Accessed on: '+ str(self.time_stamp))

    # Add function to connect to db and save data
    def init_db(self, name = 'database'):
        self.engine = create_engine('sqlite:///database/'+name+'.db', echo=False)
        # return self.engine

    # Additional functionality for same kind of data.
    # Just give in the number of points to be shown
    # def display(node, engine):
    #     # Initialise page
    #     node.init_page()
    #     show_data = st.empty()
    #     while True:
    #         query = "SELECT * FROM register_data ORDER BY time_stamp DESC LIMIT 50"
    #         df = pd.read_sql(query, con=engine)
    #         # show_data.dataframe(df)
    #         show_data.line_chart(df[["reg_plc_health"]])
