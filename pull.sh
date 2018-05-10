#!/bin/bash
scancel -u sgm400
rm out* core*
git pull origin master
make
./pull1_1.sh
./pull1_2.sh
./pull1_3.sh
./pull1_4.sh
./pull2_1.sh
./pull2_2.sh
./pull2_3.sh
./pull2_4.sh
./pull3_1.sh
./pull3_2.sh
./pull3_3.sh
./pull3_4.sh

