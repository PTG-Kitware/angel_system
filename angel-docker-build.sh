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

Build the PTG ANGEL system docker container images.

Options:
  -h | --help   Display this message.
  --force       Force image building regardless of workspace hygiene.f
"
}

function log()
{
  >&2 echo "$@"
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
      log "Forcing build regardless of workspace hygiene."
      shift
      FORCE_BUILD=1
      ;;
    *)  # anything else
      dc_forward_params+=("$1")
      shift
  esac
done

# Check if there are modified or "new" files in the workspace.
# NODE: `warn_build_spaces` should be expanded if more things start to become
#       part of docker builds. Maybe read this in from somewhere else instead
#       of encoding here?
warn_build_spaces=(
  "${SCRIPT_DIR}/ros"
  "${SCRIPT_DIR}/docker"
  "${SCRIPT_DIR}/pyproject.toml"
  "${SCRIPT_DIR}/poetry.lock"
)
git_status="$(git status --porcelain "${warn_build_spaces[@]}")"
# Check if there are ignored files in the workspace that should not be there.
git_clean_dr="$(git clean "${warn_build_spaces[@]}" -xdn)"
if [[ -n "${git_status}" ]] || [[ -n "${git_clean_dr}" ]]
then
  log "WARNING: Docker/ROS workspace subtree is modified and/or un-clean."
  if [[ -n "${git_status}" ]]
  then
    log "WARNING: -- There are modified / new files (check git status)."
  fi
  if [[ -n "${git_clean_dr}" ]]
  then
    log "WARNING: -- There are unexpected ignored files (check git clean -xdn)."
  fi
  if [[ -n "$FORCE_BUILD" ]]
  then
    log "WARNING: Force enabled, building anyway."
  else
    log "ERROR: Refusing to build images."
    exit 1
  fi
fi

if [[ "${#dc_forward_params[@]}" -gt 0 ]]
then
  # shellcheck disable=SC2145
  log "Forwarding to docker-compose: ${dc_forward_params[@]}"
fi

docker-compose \
  --env-file "$SCRIPT_DIR"/docker/.env \
  -f "$SCRIPT_DIR"/docker/docker-compose.yml \
  --profile build-only \
  build "$@"
