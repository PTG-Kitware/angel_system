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


class InvalidSkill(Exception):
    pass


def validate_message(message: str) -> None:
    """
    makes sure message is in valid format, and raises an exception if it is not
    PARAM: message - str
    """
    if message == "raise no response error":  # to demonstrate timeout
        print("sleeping")
        time.sleep(8)
    if not any("skill" and s_d in message.lower() for s_d in ["started", "done"]):
        if not message.lower() in utils.valid_messages:
            raise InvalidMessage(
                f"Invalid message, message must be in [{', '.join(utils.valid_messages)}], "
                f"or `skill XX started/stopped`"
            )
    elif not any(skill in message.lower() for skill in utils.skills):
        raise InvalidSkill(
            f"Invalid skill, skill must be one of [{', '.join(utils.skills)}]"
        )
    else:
        return


def generate_response(message: str) -> str:
    try:
        validate_message(message)
        return "OK"
    except InvalidMessage:
        return (
            f"Invalid message, message must be in [{', '.join(utils.valid_messages)}], "
            f"or `skill XX started/stopped`"
        )
    except InvalidSkill:
        return f"Invalid skill, skill must be one of [{', '.join(utils.skills)}]"
    except Exception as e:
        traceback_str = "".join(traceback.format_tb(e.__traceback__))
        return f"Client error {traceback_str}"


def talk_to_server(address: str, name: str, tmux_ctrl: TmuxController) -> None:
    """
    Summary
    Connects to the server and sends a series of messages.
    Expects a confirmation message for each message sent.

    PARAMS:
    address: [String] The socket address to connect to
    name: [String] Client name?
    tmux_ctrl: [TmuxController] Controls starting and stopping the tmux config.
    """
    context = zmq.Context()  # Get zmq context to make socket
    print(f"Connecting to server at {address}")
    # create socket from context. N.B., socket must be dealer type
    # other socket message patterns will cause errors
    socket = context.socket(zmq.DEALER)

    socket.connect(address)  # Connect to the server
    print(f"Connected to {address}")
    socket.send_string(
        f"{name}:OK", flags=zmq.DONTWAIT
    )  # Let server know client is listening

    # this loop should be modified and incorporated into client code
    # so it can listen to and respond to the server
    while True:
        # monitor the socket to see if a message is waiting, N.B, if you don't
        # include a timeout, poll WILL block
        if socket.poll(timeout=500):
            response = socket.recv_string()  # receive message
            print(f"{name}: Received message: {response}")

            if "started" in response:
                # Starting a new skill, so extract the skill name and tell the
                # tmux controller to start it.
                skill = response.replace("skill ", "").replace(" started", "")
                tmux_ctrl.start(skill)
            elif "done" in response:
                # Stopping a skill, so extract the skill name and tell the tmux
                # controller to stop it.
                skill = response.replace("skill ", "").replace(" done", "")
                tmux_ctrl.stop(skill)
            elif response == "stop":
                # Stop all tmux configurations
                print("Stopping all tmux sessions")
                tmux_ctrl.stop_all_tmux_sessions()

            # validate the message and form response
            return_message = generate_response(response)
            # send the response to the server
            # n.b. server expects string in form `name:message`
            socket.send_string(f"{name}:{return_message}", flags=zmq.DONTWAIT)
        time.sleep(0.5)
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
        help="Identifier for this client",
    )
    parser.add_argument(
        "--skill-config",
        type=str,
        nargs=2,
        metavar=("SKILL_NAME", "CONFIG_NAME"),
        help="Mapping from skill to tmuxinator configuration",
        action="append",
    )
    args = parser.parse_args()

    address = args.address
    name = args.name
    skill_cfg = args.skill_config

    if skill_cfg is None:
        print("Please provide skill to config map with `--skill-config` argument")
    else:
        skill_cfg_map = {}
        for s in skill_cfg:
            skill_cfg_map[s[0]] = s[1]

        tmux_ctrl = TmuxController(skill_cfg_map)
        try:
            talk_to_server(address, name, tmux_ctrl)
        except (Exception, KeyboardInterrupt) as ex:
            print(f"\nCaptured exception {type(ex).__name__}: {str(ex)}")
            print("Shutting down any sessions...")
            tmux_ctrl.stop_all_tmux_sessions()
            print("Shutting down any sessions... Done")
            raise
    print("End of main")
