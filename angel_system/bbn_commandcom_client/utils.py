###############################################################################
# Copyright (c) 2023 Raytheon BBN Technologies Corp.                          #
# 5775 Wayzata Boulevard, Suite 630                                           #
# Minneapolis, MN 55416                                                       #
# (952) 545-5713                                                              #
###############################################################################

import logging
import logging.config
import logging.handlers
import re

skills = ["m1","m2","m3","m5","r18"]
valid_messages = ["start", "stop", "pause"]

def validate_server_message(message:str) -> bool:
    """
    makes sure message is in valid format, and raises an exception if it is not
    PARAM: message - str
    """
    if not any("skill" and s_d in message.lower() for s_d in ['started', 'done']):
        if not message.lower() in valid_messages:
            return False
    elif not any(skill in message.lower() for skill in skills):
        return False
    else:
        return True
    
def logger_thread(q) -> None:
    """
    listener object that writes to log
    params: q - multiprocessing queue
    """
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)

def get_socket_address(address:str, increase:int=0) -> str:
    """
    Params:
        address: string (e.g. tcp://*:5555)
        increase: int number to increase address port by
    """
    ad = re.sub(r'[0-9]+$',
            lambda x: f"{str(int(x.group())+increase).zfill(len(x.group()))}",address)
    return ad