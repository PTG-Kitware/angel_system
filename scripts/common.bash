# Common functions and definitions for use in bash scripts.


# Logging function to output strings to STDERR.
function log()
{
  >&2 echo "$@"
}


# Define to a variable of the given name an array composed of the appropriate
# CLI arguments to invoke docker compose for the current system, if able at
# all.
#
# If a docker compose tool cannot be identified, this function returns code 1.
#
# This should generally be called like:
#     get_docker_compose_cmd DC_CMD
# Which results in "DC_CMD" being defined as an array in the calling context,
# viewable like:
#     echo "${DC_CMD[@]}"
#
function get_docker_compose_cmd()
{
  EXPORT_VAR_NAME="$1"
  if [[ -z "$EXPORT_VAR_NAME" ]]
  then
    log "[ERROR] No export variable name provided as the first positional argument."
    return 1
  fi
  # Check for v1 docker-compose tool, otherwise try to make use of v2
  # docker-compose plugin
  if ( docker compose >/dev/null 2>&1 )
  then
    log "[INFO] Using v2 docker compose plugin"
    EVAL_STR="${EXPORT_VAR_NAME}=( docker compose )"
  elif ( command -v docker-compose >/dev/null 2>&1 )
  then
    log "[INFO] Using v1 docker-compose python tool"
    EVAL_STR="${EXPORT_VAR_NAME}=( docker-compose )"
  else
    log "[ERROR] No docker compose functionality found on the system."
    return 1
  fi
  eval "${EVAL_STR}"
}
