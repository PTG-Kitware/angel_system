#!/bin/bash
#
# Build the docker images.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
pushd "$SCRIPT_DIR"

function usage()
{
  echo "
Usage: $0 [-h|--help] [--force]

Options:
  -h | --help   Display this message.
  --force       Force image building regardless of workspace hygiene.f
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
    --force)
      echo "Forcing build regardless of workspace hygiene."
      shift
      FORCE_BUILD=1
      ;;
    *)  # anything else
      dc_forward_params+=("$1")
      shift
  esac
done

git_status="$(git status --porcelain "${SCRIPT_DIR}/ros")"
if [[ -n "$git_status" ]]
then
  echo "WARNING: ROS workspace subtree is not clean."
  if [[ -n "$FORCE_BUILD" ]]
  then
    echo "WARNING: Force enabled, building anyway."
  else
    echo "ERROR: Refusing to build images."
    exit 1
  fi
fi

if [[ "${#dc_forward_params[@]}" -gt 0 ]]
then
  # shellcheck disable=SC2145
  echo "Forwarding to docker-compose: ${dc_forward_params[@]}"
fi

docker-compose \
  --env-file "$SCRIPT_DIR"/docker/.env \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  --profile build-only \
  build "$@"
