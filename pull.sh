#!/bin/bash

rm out* core*
git pull origin master
<<<<<<< HEAD
make
./run_job.sh 5
=======
./pull1.sh
./pull2.sh
./pull3.sh
>>>>>>> d4572b09cc0850ab5f48a7634ccedc4016e2c038
