============================
Integration Testing with BBN
============================

Testing BBNUpdate Message to YAML Transmission
==============================================
Relevant Node: ``ros2 run bbn_integration ZmqIntegrationClient``

Relevant Source Files:
* ``ros/bbn_integration/src/nodes/zmq_integration_client.cxx``
* ``ros/bbn_integration/src/ros_to_yaml.cxx``

The files contained here are manually crafted YAML serializations of the ROS2
BBNUpdate message format.
These can be emitted as ROS2 messages via ``ros2 topic pub ...`` to test the
functionality of our ``ZmqIntegrationClient`` converter node defined in this
package.

How to publish message from YAML file
-------------------------------------
Example Message YAML serializations:
* ``ros/bbn_integration/example_files/example_output.yml``
* ``ros/bbn_integration/example_files/example_output_seq_1.yml``
* ``ros/bbn_integration/example_files/example_output_seq_2.yml``
* ``ros/bbn_integration/example_files/example_output_seq_3.yml``

The ``*_seq_*.yml`` files are intended to represent a sequence of BBNUpdate
messages showing the current step changing.
These are useful to send in succession to BBN's own service in order to
validate if our output YAML format is valid and that they are appropriately
seeing the expected content changes.

Example CLI to manually publish a BBNUpdate message from a YAML file:

.. code-block:: bash

    ros2 topic pub --once \
        /BBNUpdates \
        bbn_integration_msgs/BBNUpdate \
        "$(cat ros/bbn_integration/example_files/example_output_seq_3.yml)"

The ``/BBNUpdates`` topic used in the example above is arbitrary and should be
replaced with whatever topic you parameterize ``-p topic_update_msg:=...`` for
the ``ZmqIntegrationClient`` node to be.
