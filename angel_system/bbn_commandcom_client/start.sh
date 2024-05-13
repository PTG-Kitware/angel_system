#!/bin/bash
set -e

BBN_COMMANDCOM_URI="tcp://128.33.193.178:5558"

./angel-workspace-shell.sh -r \
  python3 "/angel_workspace/angel_system/bbn_commandcom_client/client.py" \
  --address "${BBN_COMMANDCOM_URI}" \
  --name kw \
  --skill-config m2  demos/medical/BBN-M2-Tourniquet \
  --skill-config m3  demos/medical/M3-BBN-integrate-Kitware \
  --skill-config r18 demos/medical/BBN-R18
