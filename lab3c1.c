#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#define H 2 
#define W 2
#define C 1

int main(int argc, char *argv[]){
    int world_rank, world_size; 
    double checksum = 0;
    double *I_sub, *O;
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Request request[world_size-1];
    MPI_Status status[world_size-1];
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
        printf("Input %d\n",world_rank);
        for(int i=0;i<C;i++){
            for(int j=0;j<H;j++) {
                for(int k=0;k<W;k++) {
                    I_sub[i*H*W+j*W+k] = world_rank + i*(j+k);
                    printf("%lf ",I_sub[i*H*W+j*W+k]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    // printf("here3\n");
    if(world_rank==0) for(int idx = 0;idx<world_size-1;idx++) MPI_Irecv(buff+idx*C*H*W, C*H*W, MPI_DOUBLE, idx+1, 123+idx+1, MPI_COMM_WORLD, &request[idx]);
    // printf("here4q\n");
    else MPI_Isend(I_sub, C*H*W, MPI_DOUBLE, 0, 123+world_rank, MPI_COMM_WORLD, &request2);
    if(world_rank==0){
        printf("Waiting to receive everything \n");
        MPI_Waitall(world_size-1, request, status); 
        printf("Received everything \n");
    } 
    else{
        printf("Waiting to send %d \n", world_rank);
        MPI_Wait(&request2, &status2); 
        printf("Sent %d \n",world_rank);
    } 
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank==0){
        for(int idx=0;idx<world_size-1;idx++){
            for(int i=0;i<C;i++){
                for(int j=0;j<W;j++) {
                    for(int k=0;k<H;k++) {
                        printf("%lf ",buff[idx*C*H*W+i*H*W+j*W+k] );
                        O[i*H*W+j*W+k] += buff[idx*C*H*W+i*H*W+j*W+k]*((double)1/(world_size-1));
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
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
    }
    MPI_Finalize();
    free(O);
    free(I_sub);
    free(buff);
}
