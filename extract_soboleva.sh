#!/usr/bin/bash

set -ex
file="$1"
test -f "$file" || (echo "File does not exist" && false)

start=4
stride=9
length=$(wc -l "$file" | cut -d' ' -f 1)
echo "length = ${length}"
length=$(( length - stride ))

file_id=1

for (( i=start; i<=length; i+=stride, file_id++ ))
do
  cmd="${i},$((i+7))p;$((i+8))q"
  echo "Running: ${cmd}"
  sed -n ${cmd} $file > "soboleva_extracted${file_id}.txt"
done
