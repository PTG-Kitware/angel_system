============
System Setup
============


Linux Machine Setup
===================
The target linux machine will need to have installed:

* ansible
* docker
* nvidia-docker2
* docker-compose

You will want to add your user to the ``docker`` group to facilitate using
docker commands.
See `Adding a User to the docker Group`_.

The network interface to which we are going to connect to the HoloLens2 should
be appropriately configured:

* IPv4 should be set to 192.168.1.100 (address), 255.255.255.0 (netmask), and
  192.168.1.1 (gateway)

To acquire the ANGEL System software and other required files:

* Clone ANGEL System repository to the desired location

  * Remember to check out submodules with ``git submodule update --init --recursive``.

* Pull docker images with ``./angel-docker-pull.sh``. Alternatively build the
  docker images using ``./angel-docker-build.sh``.

* Provision model files (see section below).

Adding a User to the docker Group
---------------------------------
When logged in to the target user, run:

.. prompt:: bash

    sudo usermod -a -G docker

Provisioning Files
------------------
External large files should be provisioned by running the ansible tool:

    ansible-playbook -i ansible/hosts.yml ansible/provision_files.yml

This may include large files for running the system, like ML model files, or
other files required for building docker images.

This provisioning may require additional configuration and variables set in
your environment in order to satisfy some permissions:

* ``GIRDER_API_KEY`` will need to be set in order to acquire protected files from
  ``data.kitware.com`` (see `Getting your Girder API Key`_ below).

The configuration that controls what is staged and where is located
in the ``ansible/roles/provision-files/vars/main.yml`` file.

Getting your Girder API Key
^^^^^^^^^^^^^^^^^^^^^^^^^^^
1) Log into your account at `data.kitware.com`_.

2) From the drop-down in the upper-right, navigate to ``My Account``.

3) Select the "API keys" tab.

4) If you do not already have an appropriate key present, press the "Create
   new key" to make one.

   a) Only the "Read-only" permissions needs to be enabled.

5) Click the "show" text to temporarily display the key text.
   Copy and paste this into your terminal as the value for the `GIRDER_API_KEY`
   environment variable.

HL2 First-time Setup
====================

* User account creation (see MS docs
  `here <https://learn.microsoft.com/en-us/hololens/hololens2-start>`__).

* Eye Calibration

* Network connection configuration

    * Connect USB-C dongle to the HL2 headset and to the PC.

    * Configure HL2 Ethernet settings to use Manual addressing, setting the
      IPv4 to 192.168.1.101 (address), 255.255.255.0 (netmask), 192.168.1.1
      (gateway).


.. _data.kitware.com: https://data.kitware.com
