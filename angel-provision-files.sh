#!/bin/bash
#
# Standard method of invoking ansible based file provisioning.
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pushd "${SCRIPT_DIR}"

ansible-playbook \
  -i ansible/hosts.yml \
  -e ansible_python_interpreter=python3 \
  ansible/provision_files.yml
