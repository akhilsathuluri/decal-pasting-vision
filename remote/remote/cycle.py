# Call model identification
# Send information to the required register
# based on the identified model

from model_identification import *
import pandas as pd
from datetime import datetime

def health(node, map):
    while True:
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

def cycle(node, map):
    while True:
        loop_number = 1
        rq = node.client.write_registers(map['reg_pi_last_loop'], loop_number, unit=node.unit)

        trigger1 = node.client.read_holding_registers(map['reg_plc_trigger1'], 1, unit=node.unit)

        # Start cycle
        if trigger1.registers[0] == 1:
            loop_number = 2
            rq = node.client.write_registers(map['reg_pi_last_loop'], loop_number, unit=node.unit)
            # Check models
            ret, identified_model = check_model()
            # Handle not being able to write register
            if ret == True:
                # Write model number register
                reg_name = 'reg_pi_model_'+str(identified_model)
                rq = node.client.write_registers(map[reg_name], 1, unit=node.unit)
            elif ret == 'ERROR':
                # To handle camera prediction errors
                rq = node.client.write_registers(map['reg_pi_error'], 1, unit=node.unit)
            # else:
            #     # To handle unknown read/write or pi errors
            #     rq = node.client.write_registers(map['reg_pi_unknown_error'], 1, unit=node.unit)

def write_to_db(node, map, engine):
    temp = map.copy()
    temp['time_stamp'] = datetime.now()
    temp = pd.DataFrame([temp], columns=temp.keys())
    while True:
        for reg in map:
            temp_reg = node.client.read_holding_registers(map[reg], 1, unit=node.unit)
            temp[reg] = temp_reg.registers[0]
        temp['time_stamp'] = datetime.now()
        temp.to_sql('decal_data', con=engine, if_exists='append')
