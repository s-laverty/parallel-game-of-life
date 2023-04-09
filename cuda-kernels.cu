#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "clockcycle.h"

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

//TODO: wrap around
//TODO: is there a faster way to count number of neighbors?
//TODO: test with double pointer allocation as in here: https://stackoverflow.com/questions/59162457/allocate-a-2d-vector-in-unified-memory-cuda-c-c
//		and make sure it's actually slower to do it like that than with the non-dynamically allocated 2D array 
//TODO: pipelined kernel that will pipeline time steps
//		each thread waits until neighbors are at same generation as itself before continuing, work on calculating multiple generations at once

/**
 * @brief Device helper function for getting data from 
 * linearized 2D array
 *
 * @param arr Array pointer
 * @param r Row index
 * @param c Column index
 * @param pitch Mem allocation pitch
 */
__device__ bool getData(bool* arr, int r, int c, int pitch){
	bool* row_start = (bool*)((char*)arr + r * pitch);
	return row_start[c];
}

/**
 * @brief Device helper function for setting data in
 * linearized 2D array
 *
 * @param arr Array pointer
 * @param r Row index
 * @param c Column index
 * @param pitch Mem allocation pitch
 * @param val Value to set
 */
__device__ void setData(bool* arr, int r, int c, int pitch, bool val){
	bool* row_start = (bool*)((char*)arr + r * pitch);
	row_start[c] = val;
}

/**
 * @brief Compute 1 time step for a given section of the grid.
 * Each thread computes 1 cell.
 *
 * @param grid Current state of grid section (input)
 * @param pitch1 Grid memory allocation pitch
 * @param next_grid Next time step state of grid section (output)
 * @param pitch2 Next_grid memory allocation pitch
 * @param width Grid width
 * @param height Grid height
 */
__global__ void compute_timestep(bool* grid, size_t pitch1, bool* next_grid, size_t pitch2, int width, int height){
  // Each thread the next state for 1 cell
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
  unsigned int col = i % width;
  unsigned int row = i / height;
  
  // count the num of alive cells surrounding 
  unsigned int surrounding_population = 0;
  if (row > 0){
	  if (col > 0 && getData(grid, row-1, col-1, pitch1) ){
		surrounding_population++;
	  }
	  if (getData(grid, row-1, col, pitch1)){
	  	surrounding_population++;
	  }
	  if (col < width && getData(grid, row-1, col+1, pitch1)){
	  	surrounding_population++;
	  }
  }
  if (col > 0 && getData(grid, row, col-1, pitch1)){
  	surrounding_population++;
  }
  if (col < width && getData(grid, row, col+1, pitch1)){
  	surrounding_population++;
  }
  if (row < height){
  	if (col > 0 && getData(grid, row+1, col-1, pitch1)){
  		surrounding_population++;
  	}
  	if (col < width && getData(grid, row+1, col+1, pitch1) ){
		surrounding_population++;
  	}
  	if(getData(grid, row+1, col, pitch1)){
  		surrounding_population++;
  	}
  } 
	
  // Set next cell state
  bool next_cell_state = (surrounding_population == 3 || (getData(grid, row, col, pitch1) && surrounding_population==2));
  setData(next_grid, row, col, pitch2, next_cell_state);
}

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