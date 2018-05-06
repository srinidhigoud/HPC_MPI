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
    double *I_sub, *O;
    MPI_Request request[world_size-1];
    MPI_Status status[world_size-1];
    MPI_Request request2;
    MPI_Status status2;
    double *buff;
    struct timeval t1, t2;
    double elapsedTime;
    // printf("%d %d\n", world_size, world_rank);
    if(world_rank==0){
        // printf("here1\n");
        buff = (double*)malloc(sizeof(double)*(world_size-1)*C*H*W);
        O = (double*)calloc(C*H*W, sizeof(double));
        // MPI_Barrier(MPI_COMM_WORLD);
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
                        // printf("%lf ", buff[idx*C*H*W + i*H*W + j*W + k] );
                        O[i*H*W + j*W + k] += buff[idx*C*H*W + i*H*W + j*W + k];
                    }
                    checksum += O[i*H*W+j*W+k] ;
                }
                // printf("\n");
            }
            // printf("\n");
        }

        gettimeofday(&t2, NULL);

        elapsedTime = t2.tv_usec - t1.tv_usec;
         printf("%4.3lf, %4.3lf\n",checksum/(world_size-1),elapsedTime/1000);

    }
    else{

        // printf("here2\n");
        I_sub = (double*)malloc(sizeof(double)*C*H*W);
        // printf("Input %d\n",world_rank);

        for(int i=0;i<C;i++){
            for(int j=0;j<H;j++) {
                for(int k=0;k<W;k++) {
                    I_sub[i*H*W+j*W+k] = world_rank + i*(j+k);
                    // printf("%lf ",I_sub[i*H*W+j*W+k]);
                }
                // printf("\n");
            }
            // printf("\n");
        }
        // MPI_Barrier(MPI_COMM_WORLD);
        MPI_Isend(I_sub, C*H*W, MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, &request2);
        // printf("Waiting to send %d \n", world_rank);
        MPI_Wait(&request2, &status2); 
        // printf("Sent %d \n",world_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("\n");
    }
    
    MPI_Finalize();

}
