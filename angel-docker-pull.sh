#!/bin/bash
#
# Pull the docker images.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"

# source common functionalities
. "${SCRIPT_DIR}/scripts/common.bash"

pushd "$SCRIPT_DIR"

function usage()
{
  echo "
Usage: $0 [-h|--help]

Pull docker images from the registry configured via the environment parameters.

Options:
  -h | --help   Display this message.
"
}

# Option parsing
dc_forward_params=()
while [[ $# -gt 0 ]]
do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)  # anything else
      dc_forward_params+=("$1")
      shift
  esac
done

if [[ "${#dc_forward_params[@]}" -gt 0 ]]
then
  # shellcheck disable=SC2145
  log "Forwarding to docker-compose: ${dc_forward_params[@]}"
fi

get_docker_compose_cmd DC_CMD

"${DC_CMD[@]}" \
  --env-file "$SCRIPT_DIR"/docker/.env \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  --profile build-only \
  pull "$@"
