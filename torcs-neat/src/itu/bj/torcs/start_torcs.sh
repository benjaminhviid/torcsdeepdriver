#!/bin/bash

while true
do
ps cax | grep torcs > /dev/null
if [ $? -ne 0 ]; then
  torcs -r ~/.torcs/config/raceman/quickrace.xml -nodamage -nolaptime
fi
done
