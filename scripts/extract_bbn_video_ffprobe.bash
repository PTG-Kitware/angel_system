#!/bin/bash
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
