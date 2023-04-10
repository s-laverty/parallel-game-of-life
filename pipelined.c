/**
 * @file pipelined.c
 * @author Allison Harry (harrya@rpi.edu)
 * @brief This file implements a "pipelined" implementation of the game of life
 * where each GPU computes 1 generation and they are pipelined to overlap in time.
 * Can compare performance with other implementations in the report
 * @version 1.0
 * @date 2023-04-10
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#include "clockcycle.h"

// these have to match the WIDTH & HEIGHT in cuda-kernels.cu
#define WIDTH 16
#define HEIGHT 6

void cuda_init(bool* grid, bool* next_grid, int my_rank);
void run_kernel(bool* grid, bool* next_grid);
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
	bool* grid = (bool*)malloc(WIDTH*HEIGHT*sizeof(bool)); //current time step grid
	bool* next_grid = (bool*)malloc(WIDTH*HEIGHT*sizeof(bool)); //next time step grid
	cuda_init(grid, next_grid, my_rank);

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

	run_kernel(grid, next_grid);

	// Print output
	printf("\nRank %d output\n", my_rank);
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			printf("%d ", next_grid[r*WIDTH + c]);
		}
		printf("\n");
	}

	free_cudamemory(grid, next_grid);

	// Finalize the MPI environment.
  	MPI_Finalize();


}