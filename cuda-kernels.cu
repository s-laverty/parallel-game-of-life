/**
 * @file cuda-kernels.cu
 * @author Allison Harry (harrya@rpi.edu)
 * @brief This file defines the game of life CUDA kernels
 * @version 1.0
 * @date 2023-04-10
 */

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "clockcycle.h"
#include "grid.h"

#define clock_frequency 512000000
#define threads_per_block 32
#define WIDTH 20
#define HEIGHT 20


// this board template is just for convenient storage of 
// initial test board, real board is "board" variable
bool board_template[10][10] = {
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

/********************************************************************************
 * CUDA Kernel/functions using GridView (used by space division implementations)
/********************************************************************************

/**
 * @brief Compute 1 time step with a 1-cell padding around the 
 * edges of the grid. NO edge wrapping.
 * Each thread computes 1 cell.
 *
 * @param grid GridView struct
 */
__global__ void compute_timestep_nowrap(GridView grid){
	// Each thread the next state for 1 cell
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

	if (i < grid.width*grid.height){
		unsigned int padded_width = grid.width + 2; // actual width including padding
		unsigned int col = i % grid.width + 1; // add 1 for padding
		unsigned int row = i / grid.width + 1; // add 1 for padding

		// count the num of alive cells surrounding 
		unsigned int surrounding_population = 
			grid.grid.data[padded_width*(row-1) + col-1] + 
		  	grid.grid.data[padded_width*(row-1) + col] + 
		  	grid.grid.data[padded_width*(row-1) + col+1] + 
		  	grid.grid.data[padded_width*row + col-1] + 
		  	grid.grid.data[padded_width*row + col+1] + 
		   	grid.grid.data[padded_width*(row+1) + col-1] + 
		  	grid.grid.data[padded_width*(row+1) + col] + 
		  	grid.grid.data[padded_width*(row+1) + col+1];
			
		// Set next cell state
		bool next_cell_state = (surrounding_population == 3 || (grid.grid.data[padded_width*row + col] && surrounding_population==2));
		grid.next_grid.data[padded_width*row + col] = next_cell_state;	
	  }
}


/**
 * @brief Calls no wrap kernel
 *
 * @param grid GridView input
 */
extern "C" void run_kernel_nowrap(GridView* grid){
	unsigned int grid_size = ceil((grid->width*grid->height) / (float)threads_per_block);
	compute_timestep_nowrap<<<grid_size, threads_per_block>>>(*grid);
	cudaDeviceSynchronize();
}


/*
 * @brief Function to initialize a GridView in CUDA memory
 *
 * @param grid Grid
 * @param my_rank Rank of MPI process
 */
extern "C" void cuda_init_gridview(GridView* grid, int my_rank){
  // set a CUDA device for each rank with minimal overlap in device usage
  int cudaDeviceCount;
  cudaError_t cE;
  if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess ){
    printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
    exit(-1);
  }
  if( (cE = cudaSetDevice( my_rank % cudaDeviceCount )) != cudaSuccess )
  {
    printf(" Unable to have rank %d set to cuda device %d, error is %d \n",my_rank, (my_rank % cudaDeviceCount), cE);
    exit(-1);
  }

  //memory allocation/initialization
  //Grid structs contain pointers that need to be copied explicitly

  cudaMallocManaged(&(grid->grid.data), grid->grid.width*grid->grid.height*sizeof(bool));
  cudaMallocManaged(&(grid->next_grid.data), grid->next_grid.width*grid->next_grid.height*sizeof(bool));

}

/*
 * @brief Function to free CUDA memory gridview version
 *
 * @param grid Current grid
 */
extern "C" void free_cudamem_gridview(GridView* grid){
	cudaFree(grid->grid.data);
	cudaFree(grid->next_grid.data);
}




/*************************************************************************
 * CUDA Kernel/functions using bool* (used by piplined implementation)
/*************************************************************************

/**
 * @brief Compute 1 time step for the ENTIRE grid input.
 * Edges wrap around 
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
		unsigned int row = i / width;
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

/**
 * @brief Function to call CUDA kernel for entire grid
 *
 * @param grid Pointer to start of the current grid
 * @param next_grid Pointer to start of the next timestep grid
 * @param width Width of entire grid
 * @param height Height of entire grid
 */
extern "C" void run_kernel(bool* grid, bool* next_grid, int width, int height){
	unsigned int grid_size = ceil((width*height) / (float)threads_per_block);
	compute_timestep<<<grid_size, threads_per_block>>>(grid, next_grid, width, height);
	cudaDeviceSynchronize();
}


/**
 * @brief Compute 1 time step for ONE SECTION OF the grid.
 * Will properly compute the output from start_row to end_row
 * of the input grid. Edges wrap around.
 *
 * @param grid Current state of grid section (input)
 * @param next_grid Next time step state of grid section (output)
 * @param width Grid width
 * @param height Grid height
 * @param start_row Index of row to start at
 * @param end_row Index of row to end at (inclusive), will wrap if end_row > height
 */
__global__ void compute_timestep_section(bool* grid, bool* next_grid, int width, int height, int start_row, int end_row){
	// Each thread the next state for 1 cell
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 
	unsigned int section_height = end_row-start_row+1;
	if (i < width*section_height){
		unsigned int col = i % width;
		unsigned int row = i / width + start_row;
		row = (row < height) ? row : row-height;
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


/**
 * @brief Function to CUDA kernel for a section of the grid
 *
 * @param grid Pointer to start of the current grid
 * @param next_grid Pointer to start of the next timestep grid
 * @param width Width of entire grid
 * @param height Height of entire grid
 * @param start_row Row index to start at (inclusive)
 * @param end_row Row index to end at (inclusive)
 */
extern "C" void run_kernel_section(bool* grid, bool* next_grid, int width, int height, int start_row, int end_row){
	unsigned int grid_size = ceil((width*(end_row-start_row+1)) / (float)threads_per_block);
	compute_timestep_section<<<grid_size, threads_per_block>>>(grid, next_grid, width, height, start_row, end_row);
	cudaDeviceSynchronize();
}


/*
 * @brief Function to initialize CUDA memory
 *
 * @param grid Current grid
 * @param next_grid Next timestep grid
 * @param my_rank Rank of MPI process
 */
extern "C" void cuda_init(bool** grid, bool** next_grid, int my_rank){
  // set a CUDA device for each rank with minimal overlap in device usage
  int cudaDeviceCount;
  cudaError_t cE;
  if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess ){
    printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
    exit(-1);
  }
  if( (cE = cudaSetDevice( my_rank % cudaDeviceCount )) != cudaSuccess )
  {
    printf(" Unable to have rank %d set to cuda device %d, error is %d \n",my_rank, (my_rank % cudaDeviceCount), cE);
    exit(-1);
  }

  //memory allocation/initialization
  cudaMallocManaged(grid, WIDTH*HEIGHT*sizeof(bool));
  cudaMallocManaged(next_grid, WIDTH*HEIGHT*sizeof(bool));
}

/*
 * @brief Function to free CUDA memory
 *
 * @param grid Current grid
 * @param next_grid Next timestep grid
 */extern "C" void free_cudamemory(bool* grid, bool* next_grid){
  cudaFree(grid);
  cudaFree(next_grid);
}


/*
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
			if (r<10 && c<10){
				board[r*WIDTH + c] =  board_template[r][c];
			}else{
				board[r*WIDTH + c] = false;
			}
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
*/