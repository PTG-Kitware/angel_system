#!/bin/bash
#
# Push the docker images.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
pushd "$SCRIPT_DIR"

function usage()
{
  echo "
Usage: $0 [-h|--help]

Push docker images to the registry configured via the environment parameters.

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
  echo "Forwarding to docker-compose: ${dc_forward_params[@]}"
fi

docker-compose \
  --env-file "$SCRIPT_DIR"/docker/.env \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  --profile build-only \
  push "$@"
