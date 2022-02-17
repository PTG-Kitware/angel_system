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
Usage: $0 [-h|--help] [SERVICE_NAME]

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
positional=()
while [[ $# -gt 0 ]]
do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)  # positional args
      positional+=("$1")
      shift
      ;;
  esac
done

SERVICE_NAME="${positional[0]:-${DEFAULT_SERVICE_NAME}}"

# make local xauth file if not already there
export XAUTH_FILEPATH="${SCRIPT_DIR}/docker/.container_xauth/local.xauth"
if [[ ! -f "$XAUTH_FILEPATH" ]]
then
  log "[INFO] Creating local xauth file: $XAUTH_FILEPATH"
  touch "$XAUTH_FILEPATH"
  xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH_FILEPATH" nmerge -
fi

docker-compose \
  --env-file "$SCRIPT_DIR"/docker/.env \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  run --rm \
  "$SERVICE_NAME" "$@"
