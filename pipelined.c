/**
 * @file pipelined.c
 * @author Allison Harry (harrya@rpi.edu)
 * @brief This file implements a "pipelined" implementation of the game of life
 * where each GPU computes 1 generation and they are pipelined to overlap in time.
 * Should compare performance with other implementations in the report
 * @version 1.0
 * @date 2023-04-10
 */
#include "pipelined.h"

//TODO: what if timesteps != # MPI ranks? should wrap around...
//TODO: test wrapping with live cells near edges
//TODO: test performance vs only GPU calls
//TODO: use same interface as simulation.c for specifying the grid / outputting to a file
//TODO: test performance to find best ROWS_PER_GPU, make it a cmd line arg?

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


void run_pipelined(int my_rank, unsigned long num_steps){
	int num_ranks;
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
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

	unsigned int iters = ceil((double)HEIGHT/ROWS_PER_GPU); //# iterations for this rank for a single generation
	int my_t = 0; // time step for this rank in this generation
	int gen = my_rank; // generation this rank is currently computing
	int wasted_time = iters-(2*num_ranks); // num wasted time slots between generations when reusing ranks
	if (wasted_time < 0) wasted_time = 0;
	int start_t = (gen/num_ranks)*wasted_time + 2*gen; //start time for this generation
	int next_rank = (my_rank+1) % num_ranks; // next rank in line, wraps around
	int prev_rank = (my_rank-1+num_ranks) % num_ranks;
	int row_start = 0; 
	int row_end = 0;
	//int prev_rank_offset = my_rank-1;
	//if (prev_rank_offset < 0) prev_rank_offset = num_ranks; //wrap around
	MPI_Status* status = NULL;
	MPI_Request req;
	for(int t = 0; t < (num_steps-1)*2 + iters; t++){
		// Calculate one section of this rank's generation and send to next rank
		// so that the next rank can get started with the next generation

		if(t >= start_t && my_t < iters){ // if this rank has something to calculate this time step

			// Receive data from previous rank
			if (gen != 0 && my_t < iters-1){
				// have an extra section to receive if we are at the first t
				if (my_t == 0){
					row_start = ((my_t)*ROWS_PER_GPU+(gen-1)) % HEIGHT;
					row_end = row_start+ROWS_PER_GPU-1;
					
					if (row_end >= HEIGHT){ //rows wrap around so it was sent in 2 chunks
						MPI_Recv(grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL,prev_rank, 0, MPI_COMM_WORLD, status);
						MPI_Recv(grid, WIDTH*((row_end-row_start)-(HEIGHT-row_start)), MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
					}
					else{ // don't have to wrap around
						MPI_Recv(grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
					}
					printf("Rank %d receives rows %d to %d from %d at t=%d\n", my_rank, row_start, row_end, prev_rank, t);
				}

				row_start = ((my_t+1)*ROWS_PER_GPU+(gen-1)) % HEIGHT;
				row_end = row_start+ROWS_PER_GPU-1;
				// may not need to calculate all ROWS_PER_GPU at the last iteration
				// if HEIGHT is not divisible by ROWS_PER_GPU
				if (my_t+1 == iters-1 && HEIGHT % ROWS_PER_GPU != 0)
					row_end = row_start+(HEIGHT % ROWS_PER_GPU)-1;
				
				if (row_end >= HEIGHT){ //rows wrap around so it was sent in 2 chunks
					MPI_Recv(grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL,prev_rank, 0, MPI_COMM_WORLD, status);
					MPI_Recv(grid, WIDTH*((row_end-row_start)-(HEIGHT-row_start)), MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
				}
				else{ // don't have to wrap around
					MPI_Recv(grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
				}
				printf("Rank %d receives rows %d to %d from %d at t=%d\n", my_rank, row_start, row_end, prev_rank, t);
			}

			// Calcuate my section 
			row_start = (my_t*ROWS_PER_GPU+gen) % HEIGHT;
			row_end = row_start+ROWS_PER_GPU-1;
			if (my_t == iters-1 && HEIGHT % ROWS_PER_GPU != 0)
				row_end = row_start+(HEIGHT % ROWS_PER_GPU)-1;
			printf("Rank %d processes rows %d to %d at t=%d for gen %d\n", my_rank, row_start, row_end, t, gen);
			run_kernel_section(grid, next_grid, WIDTH, HEIGHT, row_start, row_end);
			
			// Send data to the next rank
			if (gen+1 < num_steps){
				printf("Rank %d sends rows %d to %d to rank %d at t=%d\n",  my_rank, row_start, row_end, next_rank, t);
				if (row_end >= HEIGHT){ //rows wrap around so must send in 2 chunks
					MPI_Isend(next_grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, &req);
					MPI_Isend(next_grid, WIDTH*((row_end-row_start)-(HEIGHT-row_start)), MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, &req);
				}
				else{ // don't have to wrap around
					MPI_Isend(next_grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, &req);
				}
			}
			my_t++;
		} 
		else if (my_t == iters){ // if we've finished calculating this generation
			gen += num_ranks; //reuse this rank in num_ranks generations from now
			if (gen < num_steps){
				my_t = 0; //restarting time when we calculate the next gen
				start_t = (gen/num_ranks)*wasted_time + 2*gen; //start time for the next generation			
			}
		}
	}
	end_cycles = clock_now();
	printf("Rank %d finished in %lf\n",  my_rank, ((double)(end_cycles-start_cycles))/clock_frequency);

	MPI_Barrier(MPI_COMM_WORLD);
	// Print output
	if (my_rank == (num_steps-1) % num_ranks){
		printf("\nRank %d output (%lu generations have passed)\n", my_rank, num_steps);
		for(int r = 0; r < HEIGHT; r++){
			for(int c = 0; c < WIDTH; c++){
				printf("%d ", next_grid[r*WIDTH + c]);
			}
			printf("\n");
		}	
	}

	free_cudamemory(grid, next_grid);
}

/*
int main(int argc, char** argv){

	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes & this process's rank
	int num_ranks;
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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

	int my_t = 0; // time step for this rank
	int row_start = 0;
	int row_end = 0;
	int prev_rank_offset = my_rank-1;
	if (prev_rank_offset < 0) prev_rank_offset = num_ranks; //wrap around
	int offset = my_rank; //row offset (ex. when offset = 3, will calculate rows 3-8 instead of 0-5)
	MPI_Status* status;
	MPI_Request* req;
	unsigned int iters = ceil((double)HEIGHT/ROWS_PER_GPU); //# iterations for this rank for a single timestep
	for(int t = 0; t < (TIMESTEPS-1)*2 + iters; t++){
		// Calculate one section of this rank's generation and send to next rank
		// so that the next rank can get started with the next generation
		if(t >= 2*my_rank && my_t < iters){
			row_start = my_t*ROWS_PER_GPU+my_rank;
			row_end = row_start+ROWS_PER_GPU-1;
			if(row_end > HEIGHT+offset-1) row_end = HEIGHT+offset-1;
			printf("Rank %d processes rows %d to %d at t=%d\n", my_rank, row_start, row_end, t);
			run_kernel_section(grid, next_grid, WIDTH, HEIGHT, row_start, row_end);
			//transfer data to the next rank
			if (my_rank+1 < num_ranks){
				printf("Rank %d sends rows %d to %d to rank %d at t=%d\n",  my_rank, row_start, row_end, my_rank+1, t);
				if (row_end >= HEIGHT && row_start >= HEIGHT){ //entirely wrap around
					row_start = row_start - HEIGHT;
					row_end = row_end - HEIGHT;
					MPI_Send(next_grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
				}
				if (row_end >= HEIGHT){ //rows wrap around so must send in 2 chunks
					MPI_Send(next_grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
					MPI_Send(next_grid, WIDTH*((row_end-row_start)-(HEIGHT-row_start)), MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
				}
				else{ // don't have to wrap around
					MPI_Send(next_grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, my_rank+1, 0, MPI_COMM_WORLD);
				}
			}
			my_t++;
		}

		// Receive data from previous rank which is computing the previous generation
		if(my_rank > 0 && t >= 2*(my_rank-1) && my_t < iters-1){
			int row_start = ROWS_PER_GPU*(t- 2*(my_rank-1)) + my_rank-1;
			int row_end = row_start+ROWS_PER_GPU-1;
			if(row_end > HEIGHT+prev_rank_offset-1) row_end = HEIGHT+prev_rank_offset-1;
			if (row_end >= HEIGHT && row_start >= HEIGHT){ //entirely wrap around
				row_start = row_start - HEIGHT;
				row_end = row_end - HEIGHT;
				MPI_Recv(grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
			}
			else if (row_end >= HEIGHT){ //rows wrap around so it was sent in 2 chunks
				MPI_Recv(grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
				MPI_Recv(grid, WIDTH*((row_end-row_start)-(HEIGHT-row_start)), MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
			}
			else{ // don't have to wrap around
				MPI_Recv(grid+WIDTH*row_start, WIDTH*ROWS_PER_GPU, MPI_C_BOOL, my_rank-1, 0, MPI_COMM_WORLD, status);
			}
			printf("Rank %d receives rows %d to %d from %d at t=%d\n", my_rank, row_start, row_end, my_rank - 1, t);
		}
		//MPI_Barrier(MPI_COMM_WORLD);
	}
	end_cycles = clock_now();
	printf("Rank %d finished in %lf\n",  my_rank, ((double)(end_cycles-start_cycles))/clock_frequency);

	MPI_Barrier(MPI_COMM_WORLD);
	// Print output
	if (my_rank == 3){
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
}*/