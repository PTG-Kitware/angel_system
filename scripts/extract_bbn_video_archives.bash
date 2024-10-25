#!/bin/bash
#
# Simple script to unarchive a bunch of ZIP file contents into subdirectories
# mirroring the name of the ZIP archive.
#
# Ensure the current directory contains the archives to be extracted, and that
# they are named in such a way as to allow for extraction into a subdirectory
# with the same name (e.g., "foo-1.0.zip" will extract into "./foo-1.0").
#
shopt -s globstar nullglob

for NAME in ./**/*.zip
do
  echo "+++ Starting $NAME +++"
  DNAME="$(dirname "$NAME")"
  BNAME="$(basename "$NAME" .zip)"
  TARGET="${DNAME}/${BNAME}"
  if [[ ! -d "${TARGET}" ]]
  then
    mkdir "$TARGET";
    unzip -d "$TARGET" "$NAME"
  fi
  echo "--- Finished $NAME ---"
done
