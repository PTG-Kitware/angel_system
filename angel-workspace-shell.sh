#!/bin/bash
#
# Launch a bash shell with the ROS2 workspace mounted.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_SERVICE_NAME="workspace-shell-dev-gpu"

function usage()
{
  echo "
Usage: $0 [-h|--help] [-s|--service=SERVICE_NAME] [--] ...

Start up a temporary container instance.
By default this will launch the ${DEFAULT_SERVICE_NAME} service.
Available services that may be specified may be found in the
./docker/docker-compose.yml configuration file.

Additional arguments, and explicitly those after a '--' are passed as command
arguments to the service run.

Options:
  -h | --help                   Display this message.
  -r | --run-setup              Start the service with the workspace setup
                                script already sourced, allowing the ready
                                invocation of workspace products.
  -s | --service=SERVICE_NAME   Explicitly use the provide service.
"
}

function log()
{
  >&2 echo "$@"
}

# Option parsing
passthrough_args=()
while [[ $# -gt 0 ]]
do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -r|--run-setup)
      shift
      export SETUP_WORKSPACE_INSTALL=true
      ;;
    -s|--service)
      # Use a specific service
      shift
      ARG_SERVICE_NAME="$1"
      if [[ "$ARG_SERVICE_NAME" = "--" ]]
      then
        log "[ERROR] Service name cannot be '--'"
        exit 1
      fi
      shift
      ;;
    --)
      # Escape the remainder of args as to be considered passthrough
      shift
      passthrough_args+=("${@}")
      break
      ;;
    *)  # passthrough_args args
      passthrough_args+=("$1")
      shift
      ;;
  esac
done

SERVICE_NAME="${ARG_SERVICE_NAME:-${DEFAULT_SERVICE_NAME}}"

# Create a permissions file for xauthority.
XAUTH_DIR="${SCRIPT_DIR}/.container_xauth"
# Exporting to be used in replacement in docker-compose file.
XAUTH_FILEPATH="$(mktemp "${XAUTH_DIR}/local-XXXXXX.xauth")"
export XAUTH_FILEPATH

# Conditionally gather if jupyter is available in the current
# environment and parameterize mounting it's runtime dir
if [[ -n "$(which jupyter)" ]]
then
  jupyter_rt_dir="$(jupyter --runtime-dir)"
  export JUPYTER_MOUNT_PARAM=(-v "${jupyter_rt_dir}:/root/.local/share/jupyter/runtime")
fi

# Print some status stuff from the ENV file we are using
#
# Encapsulating the source in a nested bash instance to not pollute the
# current env with the contents of the file.
ENV_FILE="${SCRIPT_DIR}/docker/.env"
if [[ -n "${PTG_TAG+x}" ]]
then
  >&2 echo "[INFO] Using container tag: ${PTG_TAG}"
else
  bash -c "\
    source \"${ENV_FILE}\";
    >&2 echo \"[INFO] Using container tag: \${PTG_TAG}\"
  "
fi

set +e
docker-compose \
  --env-file "$ENV_FILE" \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  run --rm \
  "${JUPYTER_MOUNT_PARAM[@]}" \
  "$SERVICE_NAME" "${passthrough_args[@]}"
DC_RUN_RET_CODE="$?"
set -e
log "[INFO] Container run exited with code: $DC_RUN_RET_CODE"

log "[INFO] Removing local xauth file."
rm "${XAUTH_FILEPATH}"

exit "$DC_RUN_RET_CODE"
