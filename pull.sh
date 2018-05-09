#!/bin/bash
scancel -u sgm400
rm out* core*
git pull origin master
make
./pull1.sh
./pull2.sh
./pull3.sh
