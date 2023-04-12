## ZMQ Command Client
A simple python client that handles communicating with a ZMQ server and
starting/stopping angel_system tmuxinator sessions.

### Starting the client
Within the angel workspace container, run the client python script with the
following command, where address is the address of the ZMQ server, name is the
name of our client node, and skill-config is the mapping from skill name to
tmuxinator configuration.
```
python3 angel_system/zmq_client/client.py --address tcp://localhost:5555 --name kw \
--skill-config m1 hl2ss_demo --skill-config m2 2023-02-test_hl2ss_full_system-coffee_allows_skip
```

Note that when the client is started, all existing tmux sessions in the angel
workspace are closed.
