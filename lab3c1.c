#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#define H 1024 
#define W 1024
#define C 3

int main(int argc, char *argv[]){
    int world_rank, world_size; 
    double checksum = 0;
    double *I_sub, *O;
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Request request[(world_size-1)];
    MPI_Status status[(world_size-1)];
     MPI_Request request2[(world_size-1)];
    MPI_Status status2[(world_size-1)];
    double **buff = (double**)malloc(sizeof(double*)*(world_size-1));
    // printf("here\n");
    printf("%d \n", world_rank);
    if(world_rank==0){
        // printf("here1\n");
        O = (double*)malloc(sizeof(double)*C*H*W);
        for(int i=0;i<C;i++){
            for(int j=0;j<H;j++) {
                for(int k=0;k<W;k++) {
                    O[i*H*W+j*W+k] = 0;
                }
            }
        }
    }
    else{
        // printf("here2\n");
        buff[world_rank-1] = (double*)malloc(sizeof(double)*C*H*W);
        I_sub = (double*)malloc(sizeof(double)*C*H*W);
        for(int i=0;i<C;i++){
            for(int j=0;j<H;j++) {
                for(int k=0;k<W;k++) {
                    I_sub[i*H*W+j*W+k] = world_rank + i*(j+k);
                }
            }
        }
    }
    // printf("here3\n");
    for(int idx=0;idx<world_size-1;idx++){
        MPI_Irecv(buff[idx], C*H*W, MPI_DOUBLE, idx+1, 123+idx+1, MPI_COMM_WORLD, &request[idx]);
    }
    // printf("here4q\n");
    MPI_Isend(I_sub, C*H*W, MPI_DOUBLE, 0, 123+world_rank, MPI_COMM_WORLD, &request2[world_rank]);
    MPI_Waitall(world_size-1, request, status); 
    MPI_Waitall(world_size-1, request2, status2); 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    for(int idx=0;idx<world_size-1;idx++){
        for(int i=0;i<C;i++){
            for(int j=0;j<W;j++) {
                for(int k=0;k<H;k++) {
                    O[i*H*W+j*W+k] += buff[idx*C*H*W+i*H*W+j*W+k]*((double)1/(world_size-1));
                }
            }
        }
    }
    for(int i=0;i<C;i++){
        for(int j=0;j<W;j++) {
            for(int k=0;k<H;k++) {
                checksum += O[i*H*W+j*W+k] ;
                printf("%lf ",O[i*H*W+j*W+k] );
            }
            printf("\n");
        }
        printf("\n\n");
    }
    printf("\n The check sum is %lf\n\n",checksum);
    free(O);
    free(I_sub);
    free(buff);
}
