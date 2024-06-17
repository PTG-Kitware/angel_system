Running Angel System with Speech QA
======================
The following are the instructions to run the angel system to track tasks whilst also interacting with GPT-4 using your voice for Question Answering (QA).

At a high level, you must:

1) start up the Angel ARUI on your HoloLens
2) start the tmuxinator script which runs the angel system and the QA nodes, for example for medical task R18:

.. code-block:: bash

   tmuxinator start demos/medical/Kitware-R18-qa

Note that for the QA portion, you'll need an OpenAI API key, which you export in the tmuxinator QA window after tmuxinator starts.

.. code-block:: bash

   export OPENAI_API_KEY=your_openapi_key
   export OPENAI_ORG_ID=your-openai-org-id

3) Run the Angel System ASR server. Installation and running are described below.

Angel System ASR Server
======================

Git-cloning the separate repo
---------------------
For now, this repo lives separately from the angel_system. Clone it by running:

.. code-block:: bash

   git clone git@github.com:ColumbiaNLP/angel-system-speech-processor.git

Installing dependencies with apt
----------------------

.. code-block:: bash

   sudo apt update && apt install -y sox ffmpeg

Running the Server
----------------------

Create conda environment
----------------------

.. code-block:: bash

   conda env create -f speech_server.yml
   conda activate speech_server

The server can then be instantiated with:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=4; python speech_server.py

(Note: you may need to remove the "export" command above, for example if you only have one GPU, so device 4 does not exist.)

Running the Client
--------------------

Create conda environment
--------------------

.. code-block:: bash

   conda env create -f speech_client.yml
   conda activate speech_client

Ensure the server is actively running on the server machine.
Also ensure the client is connected to a microphone peripheral.
This script will indicate when recording has begun. Otherwise, you can
optionally pass in a prerecorded file using the `-f/--file` flag.

.. code-block:: bash

   python speech_client.py --asr/--vd

