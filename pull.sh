#!/bin/bash

rm out* core*
git pull origin master
./pull1.sh
./pull2.sh
./pull3.sh