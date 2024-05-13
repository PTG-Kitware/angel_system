#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BBN_COMMANDCOM_URI="tcp://128.33.193.178:5558"

# We know that the workspace-shell script is two levels above here.
"${SCRIPT_DIR}"/../../angel-workspace-shell.sh -r \
  python3 "/angel_workspace/angel_system/bbn_commandcom_client/client.py" \
  --address "${BBN_COMMANDCOM_URI}" \
  --name kw \
  --skill-config m2  demos/medical/BBN-M2 \
  --skill-config m3  demos/medical/BBN-M3 \
  --skill-config m5  demos/medical/BBN-M5 \
  --skill-config r18 demos/medical/BBN-R18
