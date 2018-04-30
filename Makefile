all: ./lab3c1 ./lab3c2

CFLAGS := -O2 -std=c99
CC :=/home/am9031/anaconda3/bin/mpicc

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f ./lab3c1 ./lab3c2
