#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#define H 1024  
#define W 1024
#define C 3

int main(int argc, char *argv[]){

    int world_rank, world_size; 
    double checksum = 0;
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double *I_send, *I_rec;
    struct timeval t1, t2;
    double elapsedTime;
    I_send = (double*)malloc(sizeof(double)*C*H*W);
    I_rec = (double*)malloc(sizeof(double)*C*H*W);

    for(int i=0;i<C;i++){
        for(int j=0;j<H;j++) {
            for(int k=0;k<W;k++) {
                I_send[i*H*W+j*W+k] = world_rank + i*(j+k);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t1, NULL);
    MPI_Allreduce(I_send, I_rec, C*H*W, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i=0;i<C;i++){
        for(int j=0;j<W;j++) {
            for(int k=0;k<H;k++) {
                checksum += I_rec[i*H*W+j*W+k];
            }
        }
    }

    gettimeofday(&t2, NULL);

    elapsedTime = t2.tv_usec - t1.tv_usec;
    if(world_rank==0) printf("\n C2 \n%4.3lf, %4.3lf\n",checksum/world_size,elapsedTime/1000);
    MPI_Finalize();

}
