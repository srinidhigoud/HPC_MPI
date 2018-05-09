#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <math.h>
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
    double *I_sub, *O;
    MPI_Request request[world_size-1];
    MPI_Status status[world_size-1];
    MPI_Request request2;
    MPI_Status status2;
    double *buff;
    struct timeval t1, t2;
    if(world_rank==0){
        buff = (double*)malloc(sizeof(double)*(world_size-1)*C*H*W);
        O = (double*)calloc(C*H*W, sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        gettimeofday(&t1, NULL);
        for(int idx = 0;idx<world_size-1;idx++) MPI_Irecv(buff+idx*C*H*W, C*H*W, MPI_DOUBLE, idx+1, 123, MPI_COMM_WORLD, &request[idx]);
        // printf("Waiting to receive everything \n");
        MPI_Waitall(world_size-1, request, status); 
        // printf("Received everything \n");
        MPI_Barrier(MPI_COMM_WORLD);
        
        for(int i=0;i<C;i++){
            for(int j=0;j<W;j++) {
                for(int k=0;k<H;k++) {
                    for(int idx=0;idx<world_size-1;idx++){
                        O[i*H*W + j*W + k] += buff[idx*C*H*W + i*H*W + j*W + k];
                    }
                    checksum += O[i*H*W+j*W+k] ;
                }
            }
        }

        gettimeofday(&t2, NULL);
        
        long totalTime = (t2.tv_sec*1e6 + t2.tv_usec) - (t1.tv_sec*1e6 + t1.tv_usec);

        printf("\n C1 \n%4.3lf,",checksum/world_size);
        fflush(stdout);
        printf("%4.3lf\n",totalTime/1000);

    }
    else{

        I_sub = (double*)malloc(sizeof(double)*C*H*W);

        for(int i=0;i<C;i++){
            for(int j=0;j<H;j++) {
                for(int k=0;k<W;k++) {
                    I_sub[i*H*W+j*W+k] = world_rank + i*(j+k);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Isend(I_sub, C*H*W, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, &request2);
        // printf("Waiting to send %d \n", world_rank);
        MPI_Wait(&request2, &status2); 
        // printf("Sent %d \n",world_rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Finalize();

}
