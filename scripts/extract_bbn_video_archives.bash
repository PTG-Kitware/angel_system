#!/bin/bash
#
# Simple script to unarchive a bunch of ZIP file contents into subdirectories
# mirroring the name of the ZIP archive.
#
# Ensure the current directory contains the archives to be extracted, and that
# they are named in such a way as to allow for extraction into a subdirectory
# with the same name (e.g., "foo-1.0.zip" will extract into "./foo-1.0").
#

for NAME in *.zip
do
  echo "+++ Starting $NAME +++"
  BNAME="$(basename "$NAME" .zip)"
  if [[ ! -d "${BNAME}" ]]
  then
    mkdir "$BNAME";
    unzip -d "$BNAME" "$NAME"
  fi
  echo "--- Finished $NAME ---"
done
