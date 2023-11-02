#!/bin/bash
set -e
# Workspace build component -- NPM component build / installation
pushd "${ANGEL_WORKSPACE_DIR}"/src/angel_utils/demo_ui
npm install
popd
pushd "${ANGEL_WORKSPACE_DIR}"/src/angel_utils/multi_task_demo_ui
npm install
popd
