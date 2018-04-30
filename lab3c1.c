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
    double ***I_sub, ***O;
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Request request[2*(world_size-1)];
    MPI_Status status[2*(world_size-1)];
    double ****buff = (double****)malloc(sizeof(double***)*(world_size-1));
    if(world_rank==0){
        for(int i=0;i<world_size-1;i++){
            buff[i] = (double***)malloc(sizeof(double**)*C);
            for(int j=0;j<C;j++) {
                buff[i][j] = (double**)malloc(sizeof(double*)*W);
                for(int k=0;k<W;k++) {
                    buff[i][j][k] = (double*)malloc(sizeof(double)*H);
                }
            }
        }
        O = (double***)malloc(sizeof(double**)*C);
        for(int i=0;i<C;i++){
            O[i] = (double**)malloc(sizeof(double*)*W);
            for(int j=0;j<W;j++) {
                O[i][j] = (double*)malloc(sizeof(double)*H);
                for(int k=0;k<H;k++) {
                    O[i][j][k] = 0;
                }
            }
        }
    }
    else{
        I_sub = (double***)malloc(sizeof(double**)*C);
        for(int i=0;i<C;i++){
            I_sub[i] = (double**)malloc(sizeof(double*)*W);
            for(int j=0;j<W;j++) {
                I_sub[i][j] = (double*)malloc(sizeof(double)*H);
                for(int k=0;k<H;k++) {
                    I_sub[i][j][k] = world_rank + i*(j+k);
                }
            }
        }
    }
    for(int idx=0;idx<world_size-1;idx++){
        MPI_Irecv(buff[idx], C*H*W, MPI_DOUBLE, idx+1, 123+idx+1, MPI_COMM_WORLD, &request[idx]);
    }
    MPI_Isend(I_sub, C*H*W, MPI_DOUBLE, 0, 123+world_rank, MPI_COMM_WORLD, &request[world_rank+world_size-2]);
    MPI_Waitall(world_size-1, request, status); 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    for(int idx=0;idx<world_size-1;idx++){
        for(int i=0;i<C;i++){
            for(int j=0;j<W;j++) {
                for(int k=0;k<H;k++) {
                    O[i][j][k] += buff[idx][i][j][k]*((double)1/(world_size-1));
                }
            }
        }
    }
    for(int i=0;i<C;i++){
        for(int j=0;j<W;j++) {
            for(int k=0;k<H;k++) {
                checksum += O[i][j][k];
                printf("%lf ",O[i][j][k]);
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
