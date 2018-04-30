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
    MPI_Request request;
    MPI_Status status;
     MPI_Request request2;
    MPI_Status status2;
    double *buff;
    // printf("here\n");
    printf("%d %d\n", world_size, world_rank);
    if(world_rank==0){
        // printf("here1\n");
        buff = (double*)malloc(sizeof(double*)*(world_size-1)*C*H*W);
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
    MPI_Irecv(buff+idx*C*H*W, C*H*W, MPI_DOUBLE, world_rank, 123+world_rank, MPI_COMM_WORLD, &request);
    // printf("here4q\n");
    MPI_Isend(I_sub, C*H*W, MPI_DOUBLE, 0, 123+world_rank, MPI_COMM_WORLD, &request2);
    MPI_Wait(&request, &status); 
    MPI_Wait(&request2, &status2); 
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
