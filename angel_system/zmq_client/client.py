###############################################################################
# Copyright (c) 2023 Raytheon BBN Technologies Corp.                          #
# 5775 Wayzata Boulevard, Suite 630                                           #
# Minneapolis, MN 55416                                                       #
# (952) 545-5713                                                              #
###############################################################################

"""
This is a modified version of the BBN example zmq client. It has been modified
to start and stop tmuxinator sessions when "start" and "stop" commands are
received from the zmq server.
"""

import argparse
import sys
import time
import traceback

import zmq

from tmux_controller import TmuxController
import utils


class InvalidMessage(Exception):
    pass


class InvalidSkill(InvalidMessage):
    pass


def validate_message(message: str) -> None:
    """
    makes sure message is in valid format, and raises an exception if it is not
    PARAM: message - str
    """
    if message == "raise no response error":  # to demonstrate timeout
        print("sleeping")
        time.sleep(8)
    if not any("skill" and s_d in message.lower() for s_d in ['started', 'done']):
        if not message.lower() in utils.valid_messages:
            raise InvalidMessage(f"Invalid message, message must be in [{', '.join(utils.valid_messages)}], or `skill XX started/stopped`")
    elif not any(skill in message.lower() for skill in utils.skills):
        raise InvalidSkill(f"Invalid skill, skill must be one of [{', '.join(utils.skills)}]")
    else:return


def generate_response(message:str) -> str:
    try:
        validate_message(message)
        return "OK"
    except InvalidMessage:
        return f"Invalid message, message must be in [{', '.join(utils.valid_messages)}], or `skill XX started/stopped`"
    except InvalidSkill:
        return f"Invalid skill, skill must be one of [{', '.join(utils.skills)}]"
    except Exception as e:
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        return f"Client error {traceback_str}"


def talk_to_server(address:str, name:str, tmux_ctrl:TmuxController) -> None:
    """
    Summary
    Connects to the server and sends a series of messages.
    Expects a confirmation message for each message sent.

    PARAMS:
    address: [String] The socket address to connect to
    name: [String] Client name?
    tmux_ctrl: [TmuxController] Controls starting and stopping the tmux config.
    """
    context = zmq.Context()    # Get zmq context to make socket
    print(f"Connecting to server at {address}")
    socket = context.socket(zmq.DEALER)         # create socket from context. N.B., socket must be dealer type
                                                # other socket message patterns will cause errors

    socket.connect(address)  # Connect to the server
    print(f"Connected to {address}")
    socket.send_string(f"{name}:OK",flags=zmq.DONTWAIT)   # Let server know client is listening

    # this loop should be modified and incorporated into client code
    # so it can listen to and respond to the server
    while (True):
        if socket.poll(timeout=2):                                # monitor the socket to see if a message is waiting, N.B, if you don't include a timeout, poll WILL block
            response = socket.recv_string()                       # receive message
            print(f"{name}: Received message: {response}")

            if response == "start":
                # Start the tmux configuration
                tmux_ctrl.start()

            elif response == "stop":
                # Stop the tmux configuration
                tmux_ctrl.stop()

            return_message = generate_response(response)          # validate the message and form response
            socket.send_string(f"{name}:{return_message}",flags=zmq.DONTWAIT)  # send the response to the server
                                                                                # n.b. server expects string in form `name:message`
        time.sleep(.5)
        continue


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        default="tcp://localhost:5555",
        help="Address of the zmq server",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
    )
    parser.add_argument(
        "--tmux_config",
        type=str,
        default="hl2ss_demo",
    )
    args = parser.parse_args()

    address = args.address
    name = args.name
    tmux_cfg = args.tmux_config

    tmux_ctrl = TmuxController(tmux_cfg)

    talk_to_server(address, name, tmux_ctrl)
