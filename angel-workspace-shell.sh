#!/bin/bash
#
# Launch a bash shell with the ROS2 workspace mounted.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/docker/.env"

DEFAULT_SERVICE_NAME="workspace-shell-dev-gpu"

function usage()
{
  echo "
Usage: $0 [-h|--help] [-s|--service SERVICE_NAME]

Start up a temporary container instance.
By default this will launch the ${DEFAULT_SERVICE_NAME} service.
Available services that may be specified may be found in the
./docker/docker-compose.yml configuration file.

SERVICE         Name of the angel-system service to start a container for.

Options:
  -h | --help   Display this message.
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
XAUTH_DIR="${SCRIPT_DIR}/docker/.container_xauth"
# Exporting to be used in replacement in docker-compose file.
XAUTH_FILEPATH="$(mktemp "${XAUTH_DIR}/local-XXXXXX.xauth")"
export XAUTH_FILEPATH
log "[INFO] Creating local xauth file: $XAUTH_FILEPATH"
touch "$XAUTH_FILEPATH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_FILEPATH" nmerge -

set +e
docker-compose \
  --env-file "$SCRIPT_DIR"/docker/.env \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  run --rm \
  "$SERVICE_NAME" "${passthrough_args[@]}"
DC_RUN_RET_CODE="$?"
set -e
log "[INFO] Container run exited with code: $DC_RUN_RET_CODE"

log "[INFO] Removing local xauth file."
rm "${XAUTH_FILEPATH}"

exit "$DC_RUN_RET_CODE"
