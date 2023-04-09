#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
//#include "clockcycle.h"

#define threads_per_block 10
#define WIDTH 10
#define HEIGHT 10

// contiguous allocation is better than double pointer 
// for copying to/from CUDA
bool board[WIDTH][HEIGHT] = {
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, true, false, false, false, false, false, false},
	{false, true, false, true, false, false, false, false, false, false},
	{false, false, true, true, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
	{false, false, false, false, false, false, false, false, false, false},
};
bool output_board[WIDTH][HEIGHT];

//TODO: get it actually working
//TODO: wrap around
//TODO: test with double pointer allocation as in here: https://stackoverflow.com/questions/59162457/allocate-a-2d-vector-in-unified-memory-cuda-c-c
//		and show that it is slower to do it like that than with the "real" 2D array
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
__global__ void compute_timestep(bool* grid, size_t pitch1, bool* next_grid, size_t pitch2, int width, int height){
  // Each thread the next state for 1 cell
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
  unsigned int col = i % width;
  unsigned int row = i / height;

	//TEST - for now, everybody toggle your spot
		bool* row_start = (bool*)((char*)grid + row * pitch1);
		bool column_element = row_start[col];
		//if (column_element == true){
		//	row_start[10000] = 10;
		//}

		bool* r_start = (bool*)((char*)next_grid + row * pitch2);
		r_start[col] = !column_element;


  /*
  // count the num of alive cells surrounding 
  // TODO: more efficient way to do this?? some way to not double-count?? ie each cell is being counted by 9 surrounding threads
  unsigned int surrounding_population = 0;
  if (row > 0){
	  if (col > 0 && grid[row-1][col-1] ){
		surrounding_population++;
	  }
	  if (grid[row-1][col]){
	  	surrounding_population++;
	  }
	  if (col < width && grid[row-1][col+1]){
	  	surrounding_population++;
	  }
  }
  if (col > 0 && grid[row][col-1]){
  	surrounding_population++;
  }
  if (col < width && grid[row][col+1]){
  	surrounding_population++;
  }
  if (row < height){
  	if (col > 0 && grid[row+1][col-1]){
  		surrounding_population++;
  	}
  	if (col < width && grid[row+1][col+1] ){
		surrounding_population++;
  	}
  	if(grid[row+1][col]){
  		surrounding_population++;
  	}
  } 

  // Set next cell state (without if-statements is faster??)
  // TODO: check whether if-statements are actually faster
  next_grid[row][col] = (surrounding_population == 2 || (grid[row][col] && surrounding_population==3));*/
}

/*
// TODO: pipelined implementation
// compute t time steps for grid - use MPI to pipeline even with several sections????
// should be faster than doing it 1 time step at a time
__global__ void compute_pipelined(int* ssgl, int* sspl, int* sssgm, int* ssspm)
{
}
*/

// temporary main for testing CUDA kernel
int main(int argc, char *argv[]){


	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			printf("%d ", board[r][c]);
		}
		printf("\n");
	}
	printf("\n");

	// Allocate memory in the GPU
	bool* dev_ptr_board;
	bool* dev_ptr_output_board;
	size_t pitch1;
	size_t pitch2;
	// cudaMallocPitch allocates pitched device memory
	cudaMallocPitch((void**)&dev_ptr_board, &pitch1, WIDTH * sizeof(bool), HEIGHT);
	cudaMallocPitch((void**)&dev_ptr_output_board, &pitch2, WIDTH * sizeof(bool), HEIGHT);
	// copy 2D arrays to pitched device memory
	cudaMemcpy2D(dev_ptr_board, pitch1, board, WIDTH*sizeof(bool), WIDTH*sizeof(bool), HEIGHT, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_ptr_output_board, pitch2, output_board, WIDTH*sizeof(bool), WIDTH*sizeof(bool), HEIGHT, cudaMemcpyHostToDevice);

	// call kernel
	unsigned int grid_size = (HEIGHT*WIDTH) / threads_per_block;
	compute_timestep<<<grid_size, threads_per_block>>>(dev_ptr_board, pitch1, dev_ptr_output_board, pitch2, WIDTH, HEIGHT);

	// copy memory back
	cudaMemcpy2D(board, WIDTH * sizeof(bool), dev_ptr_board, pitch1, WIDTH * sizeof(bool), HEIGHT, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(output_board, WIDTH * sizeof(bool), dev_ptr_output_board, pitch1, WIDTH * sizeof(bool), HEIGHT, cudaMemcpyDeviceToHost);

	// debugging - print output
	printf("\n");
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			printf("%d ", output_board[r][c]);
		}
		printf("\n");
	}
}