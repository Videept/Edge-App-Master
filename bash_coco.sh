#!/bin/bash

array=(25 50  75 100 )
#array=(25)
size=512
protos=(tcp udp)
protos=(udp)
for proto in ${protos[@]}; do
for i in "${array[@]}"
do
  echo $proto $i $size
  python runtest_coco.py $proto $i $size > out_$proto\_$i\_$size\.txt
  mv results_client results_client_$proto\_$i\_$size\.txt
  mv results_server results_server_$proto\_$i\_$size\.txt
  mv summary.txt summary_$proto\_$i\_$size\.txt
done
done
