#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "clockcycle.h"
#include "grid.h"

void run_kernel_nowrap(GridView* grid);
void cuda_init_gridview(GridView* grid, int my_rank);
void free_cudamem_gridview(GridView* grid);

bool grid_template[10][10] = {
	{true, true, false, false, false, false, false, false, false, false},
	{true, false, false, false, false, false, false, false, false, false},
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
	//Initialize test GridView on both device & CUDA
	GridView grid; 
	grid.width = 8;
	grid.height = 8;

	Grid curr;
	curr.width = 10;
	curr.height = 10;

	Grid next;
	next.width = 10;
	next.height = 10;

	grid.grid = curr;
	grid.next_grid = next;

	cuda_init_gridview(&grid, 0); //CUDA initialize


	// fill initial values
	for(int r = 0; r < 10; r++){
		for(int c = 0; c < 10; c++){
			grid.grid.data[r*10 + c] = grid_template[r][c];
			grid.next_grid.data[r*10 + c] = grid_template[r][c];
			
			printf("%d ", grid.grid.data[r*10 + c]);
		}
		printf("\n");
	}


	// call kernel
	run_kernel_nowrap(&grid);

	// print output
	printf("\n");
	for(int r = 0; r < 10; r++){
		for(int c = 0; c < 10; c++){			
			printf("%d ", grid.next_grid.data[r*10 + c]);
		}
		printf("\n");
	}

	// i'm very responsible
	free_cudamem_gridview(&grid);
}