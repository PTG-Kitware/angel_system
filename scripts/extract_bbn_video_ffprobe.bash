#!/bin/bash
#
# This is a convenience script to invoke ffprobe on all of the MP4 files found
# under a directory, and then log the output of ffprobe in a symmetric location
# in a working directory. This is due to expecting the MP4 source location to
# be a read-only "golden" space that we don't want to write computed files to.
#
# Example:
#   $ export DATAPATH=/path/to/data
#   $ export WORKDIR=/path/to/workdir
#   $ ./extract_bbn_video_ffprobe.bash
#
shopt -s globstar nullglob

DATAPATH="${DATAPATH:-"/home/local/KHQ/paul.tunison/data/darpa-ptg/bbn_data/lab_data-golden"}"
WORKDIR="${WORKDIR:-"/home/local/KHQ/paul.tunison/data/darpa-ptg/bbn_data/lab_data-working"}"

for F in "${DATAPATH}"/**/*.mp4
do
  echo $F
  F_rel="$(realpath --relative-to="${DATAPATH}" "$F")"
  output_dirpath="$(realpath -m "${WORKDIR}/$(dirname "$F_rel")")"
  output_filepath="${output_dirpath}/$(basename "$F" .mp4).probe.txt"
  mkdir -p "${output_dirpath}"
  if [[ ! -f "${output_filepath}" ]]
  then
    ffprobe "${F}" 2>&1 | tee "${output_filepath}"
  fi
done

# Collect all probed video streams metadata summary for resolution and fps info
# $> grep -rin "Stream.*: Video" ../lab_data-working/ >../probe_summary.txt
