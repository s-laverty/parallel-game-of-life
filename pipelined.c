/**
 * @file pipelined.c
 * @author Allison Harry (harrya@rpi.edu)
 * @brief This file implements a "pipelined" implementation of the game of life
 * where each GPU computes 1 generation and they are pipelined to overlap in time.
 * Should compare performance with other implementations in the report
 * @version 1.0
 * @date 2023-04-10
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#include "clockcycle.h"

//TODO: what if HEIGHT is not divisible by ROWS_PER_GPU?
//TODO: what if TIMESTEPS != # MPI ranks? should wrap around...
//TODO: test wrapping with live cells near edges
//TODO: test performance vs only GPU calls

// these have to match the WIDTH & HEIGHT in cuda-kernels.cu
#define WIDTH 20
#define HEIGHT 20
#define TIMESTEPS 4 // number of generations to calculate
#define ROWS_PER_GPU 5 //number of rows computed by each GPU every time step
#define clock_frequency 512000000

void cuda_init(bool** grid, bool** next_grid, int my_rank);
void run_kernel(bool* grid, bool* next_grid, int width, int height);
void run_kernel_section(bool* grid, bool* next_grid, int width, int height, int start_row, int end_row);
void free_cudamemory(bool* grid, bool* next_grid);

// this board template is just for convenient storage of 
// initial test board
bool grid_template[10][10] = {
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, true, false, false, false, false, false, false},
	{false, true, false, true, false, false, false, false, false, false},
	{false, false, true, true, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, true},
	{false, false, false, false, false, false, false, true, false, true},
	{false, false, false, false, false, false, false, false, true, true},
};

int main(int argc, char** argv){

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes & this process's rank
	int num_ranks;
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//for now, just have each rank run the same thing so we 
	//can make sure the setup is correct

	// Allocate memory
	bool* grid; //current time step grid
	bool* next_grid; //next time step grid
	cuda_init(&grid, &next_grid, my_rank);

	// Initialize & print test memory
	if(my_rank == 0) printf("INPUT:\n");
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			if (r<10 && c<10){
				grid[r*WIDTH + c] =  grid_template[r][c];
			}else{
				grid[r*WIDTH + c] = false;
			}
			next_grid[r*WIDTH + c] = false;
			if(my_rank == 0) printf("%d ", grid[r*WIDTH + c]);
		}
		if(my_rank == 0) printf("\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);

	unsigned long long start_cycles=clock_now(); // dummy clock reads to init
  	unsigned long long end_cycles=clock_now();   // dummy clock reads to init
	start_cycles = clock_now();

	// Synchronize between each time step
	int my_t = 0; // time step for this rank
	int row_start = 0;
	MPI_Status* status;
	MPI_Request* req;
	for(int t = 0; t < (TIMESTEPS-1)*2 + HEIGHT/ROWS_PER_GPU; t++){
		// Calculate one section of this rank's generation and send to next rank
		// so that the next rank can get started with the next generation
		if(t >= 2*my_rank && my_t < HEIGHT/ROWS_PER_GPU){
			row_start = my_t*ROWS_PER_GPU+my_rank;
			printf("Rank %d processes rows %d to %d at t=%d\n", my_rank, row_start, row_start+ROWS_PER_GPU-1, t);
			run_kernel_section(grid, next_grid, WIDTH, HEIGHT, row_start, row_start+ROWS_PER_GPU-1);
			//transfer data to the next rank
			if (my_rank+1 < num_ranks){
				printf("Rank %d sends data to %d starting at %d at t=%d\n", my_rank, my_rank+1, row_start, t);
				if (row_start+ROWS_PER_GPU-1 >= HEIGHT){ //rows wrap around so must send in 2 chunks
					MPI_Send(next_grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
					MPI_Send(next_grid, WIDTH*(ROWS_PER_GPU-(HEIGHT-row_start)), MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
				}
				else{ // don't have to wrap around
					MPI_Send(next_grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
				}
			}
			my_t++;
		}

		// Receive data from previous rank which is computing the previous generation
		if(my_rank > 0 && t >= 2*(my_rank-1) && my_t < HEIGHT/ROWS_PER_GPU-1){
			int row_start = ROWS_PER_GPU*(t- 2*(my_rank-1)) + my_rank-1;
			if (row_start+ROWS_PER_GPU-1 >= HEIGHT){ //rows wrap around so it was sent in 2 chunks
				MPI_Recv(next_grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
				MPI_Recv(next_grid, WIDTH*(ROWS_PER_GPU-(HEIGHT-row_start)), MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
			}
			else{ // don't have to wrap around
				MPI_Recv(grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
			}
			printf("Rank %d receives data from %d starting at %d at t=%d\n", my_rank, my_rank - 1, row_start, t);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	end_cycles = clock_now();
	printf("Rank %d finished in %lf\n",  my_rank, ((double)(end_cycles-start_cycles))/clock_frequency);

	MPI_Barrier(MPI_COMM_WORLD);
	// Print output
	if (my_rank == 1){
		printf("\nRank %d output (%d generations have passed)\n", my_rank, my_rank+1);
		for(int r = 0; r < HEIGHT; r++){
			for(int c = 0; c < WIDTH; c++){
				printf("%d ", next_grid[r*WIDTH + c]);
			}
			printf("\n");
		}	
	}


	free_cudamemory(grid, next_grid);

	// Finalize the MPI environment.
  	MPI_Finalize();


}