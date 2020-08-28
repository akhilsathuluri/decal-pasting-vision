import pandas as pd
from datetime import datetime
# from node import node
from model_identification import check_model

# Need to cache both the functions

# @st.cache
def health(node, map):
    while True:
        print('health')
        loop_number = 0
        rq = node.client.write_registers(map['reg_pi_last_loop'], loop_number, unit=node.unit)
        # Read write health bits
        heartbeat_read = node.client.read_holding_registers(map['reg_plc_health'], 1, unit=node.unit)
        heartbeat_write = node.client.write_registers(map['reg_pi_health'], heartbeat_read.registers[0], unit=node.unit)

        print(heartbeat_read.registers[0])

        # Read write ready to trigger bits
        trigger_read = node.client.read_holding_registers(map['reg_plc_ready_to_trigger'], 1, unit=node.unit)
        trigger_write = node.client.write_registers(map['reg_pi_ready_for_trigger'], trigger_read.registers[0], unit=node.unit)
        # Check if PLC is reset in between
        reset = node.client.read_holding_registers(map['reg_plc_reset'], 1, unit=node.unit)
        if reset.registers[0] == 1:
            # Reset entire memory block data
            rq = node.client.write_registers(start_register, [0]*block_length, unit=node.unit)
        else:
            pass

# @st.cache(hash_funcs={node.Node: True})
def cycle(node, map, cam):
    while True:
        loop_number = 1
        rq = node.client.write_registers(map['reg_pi_last_loop'], loop_number, unit=node.unit)

        trigger1 = node.client.read_holding_registers(map['reg_plc_trigger1'], 1, unit=node.unit)

        # Trigger 1 only after component seat check is verified (handled by PLC)
        # Start cycle
        if trigger1.registers[0] == 1:
            loop_number = 2
            rq = node.client.write_registers(map['reg_pi_last_loop'], loop_number, unit=node.unit)
            # Check models
            ret, pred, frame = check_model(cam)
            # Handle not being able to write register
            if pred != 'None':
                # Write model verify register
                rq = node.client.write_registers(map['reg_pi_{}'.format(pred)], 1, unit=node.unit)
            elif pred == 'None':
                pass
            elif pred == 'ERROR':
                # To handle camera prediction errors
                rq = node.client.write_registers(map['reg_pi_error'], 1, unit=node.unit)
            else:
                # To handle unknown read/write or pi errors
                rq = node.client.write_registers(map['reg_pi_unknown_error'], 1, unit=node.unit)

def write_to_db(node, map, engine):
    temp = map.copy()
    temp['time_stamp'] = datetime.now()
    temp = pd.DataFrame([temp], columns=temp.keys())
    while True:
        print('to_db')
        for reg in map:
            temp_reg = node.client.read_holding_registers(map[reg], 1, unit=node.unit)
            temp[reg] = temp_reg.registers[0]
        temp['time_stamp'] = datetime.now()
        temp.to_sql('register_data', con=engine, if_exists='append')
