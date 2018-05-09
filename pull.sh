#!/bin/bash

rm out* core*
git pull origin master
make
./run_job.sh 5
