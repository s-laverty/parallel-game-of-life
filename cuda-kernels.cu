#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "clockcycle.h"

#define clock_frequency 512000000
#define threads_per_block 12
#define WIDTH 10
#define HEIGHT 10


// this board template is just for convenient storage of 
// initial test board, real board is "board" variable
bool board_template[WIDTH][HEIGHT] = {
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


//TODO: pipelined kernel that will pipeline time steps
//		each thread waits until neighbors are at same generation as itself before continuing, work on calculating multiple generations at once

/**
 * @brief Compute 1 time step for a given section of the grid.
 * Each thread computes 1 cell.
 *
 * @param grid Current state of grid section (input)
 * @param next_grid Next time step state of grid section (output)
 * @param width Grid width
 * @param height Grid height
 */
__global__ void compute_timestep(bool* grid, bool* next_grid, int width, int height){
	// Each thread the next state for 1 cell
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
	if (i < width*height){
		unsigned int col = i % width;
		unsigned int row = i / height;
		// calculate wrapped-around values
		unsigned int col_plus1 = (col < width-1) ? col+1 : 0;
		unsigned int col_minus1 = (col > 0) ? col-1 : width-1;
		unsigned int row_plus1 = (row < height-1) ? row+1 : 0;
		unsigned int row_minus1 = (row > 0) ? row-1 : height-1;

		// count the num of alive cells surrounding 
		unsigned int surrounding_population = 
			grid[width*row_minus1 + col_minus1] + 
		  	grid[width*row_minus1 + col] + 
		  	grid[width*row_minus1 + col_plus1] + 
		  	grid[width*row + col_minus1] + 
		  	grid[width*row + col_plus1] + 
		   	grid[width*row_plus1 + col_minus1] + 
		  	grid[width*row_plus1 + col] + 
		  	grid[width*row_plus1 + col_plus1];
			
		// Set next cell state
		bool next_cell_state = (surrounding_population == 3 || (grid[width*row + col] && surrounding_population==2));
		next_grid[width*row + col] = next_cell_state;	
	  }
}

// temporary main for testing CUDA kernel
int main(int argc, char *argv[]){

	// Allocate memory
	bool* board = (bool*)malloc(WIDTH*HEIGHT*sizeof(bool));
	bool* output_board = (bool*)malloc(WIDTH*HEIGHT*sizeof(bool));
	cudaMallocManaged(&board, WIDTH*HEIGHT*sizeof(bool));
	cudaMallocManaged(&output_board, WIDTH*HEIGHT*sizeof(bool));


	// Fill out board 
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			board[r*WIDTH + c] =  board_template[r][c];
			output_board[r*WIDTH + c] = false;
			printf("%d ", board[r*WIDTH + c]);
		}
		printf("\n");
	}
	printf("\n");

	unsigned long long start_cycles=clock_now(); // dummy clock reads to init
  	unsigned long long end_cycles=clock_now();   // dummy clock reads to init
	start_cycles = clock_now();

	// call kernel
	unsigned int grid_size = ceil((HEIGHT*WIDTH) / (float)threads_per_block);
	compute_timestep<<<grid_size, threads_per_block>>>(board, output_board, WIDTH, HEIGHT);
	cudaDeviceSynchronize();

	end_cycles = clock_now();
	printf("Finished in %lf\n",  ((double)(end_cycles-start_cycles))/clock_frequency);

	// print output
	printf("\n");
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			printf("%d ", output_board[r*WIDTH + c]);
		}
		printf("\n");
	}

	//free memory
	cudaFree(board);
	cudaFree(output_board);
}