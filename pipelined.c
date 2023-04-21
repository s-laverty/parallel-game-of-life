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

//TODO: use same interface as simulation.c for specifying the grid / outputting to a file
//TODO: test performance to find best ROWS_PER_GPU, make it a cmd line arg?

void cuda_init(bool** grid, bool** next_grid,  unsigned long width,  unsigned long height, int my_rank);
//void run_kernel(bool* grid, bool* next_grid,  unsigned long width,  unsigned long height);
void run_kernel_section(bool* grid, bool* next_grid, unsigned long width, unsigned long height, int start_row, int end_row);
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


//file loading for testing purposes
void load_input_file(char* filename, bool* grid, unsigned long HEIGHT, unsigned long WIDTH);
void load_hardcode_config(const char* hc_config, bool* grid, unsigned long width, unsigned long height);

void run_pipelined(int my_rank, unsigned long HEIGHT, unsigned long WIDTH, unsigned long num_steps, const char* hc_config){
	int num_ranks;
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	// Allocate memory
	bool* grid; //current time step grid
	bool* next_grid; //next time step grid
	cuda_init(&grid, &next_grid, WIDTH, HEIGHT, my_rank);

	// load in initial data for rank 0
	if (my_rank == 0){
		//char filename[] = INPUT_FILE; 
		//load_input_file(filename, grid, HEIGHT, WIDTH);	
		load_hardcode_config(hc_config, grid, WIDTH, HEIGHT);

	}

	MPI_Barrier(MPI_COMM_WORLD);

	// start clock
	unsigned long long start_cycles=clock_now(); // dummy clock reads to init
  	unsigned long long end_cycles=clock_now();   // dummy clock reads to init
	start_cycles = clock_now();

	// begin game of life simulation
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
	MPI_Status* status = NULL;
	MPI_Request req;
	for(int t = 0; t < (num_steps-1)*2 + iters; t++){
		// if there are extra ranks, we don't need them to do anything
		if (my_rank > num_steps){
			break;
		}

		// Calculate one section of this rank's generation and send to next rank
		// so that the next rank can get started with the next generation

		// if this rank has something to calculate this time step
		if(t >= start_t && my_t < iters){ 

			// Receive data from previous rank
			if (gen != 0 && my_t < iters-1){
				// have an extra section to receive if we are at the first t
				if (my_t == 0){
					row_start = ((my_t)*ROWS_PER_GPU+(gen-1)) % HEIGHT;
					row_end = row_start+ROWS_PER_GPU-1;
					
					if (row_end >= HEIGHT){ //rows wrap around so it was sent in 2 chunks
						MPI_Recv(grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL,prev_rank, 0, MPI_COMM_WORLD, status);
						MPI_Recv(grid, WIDTH*((row_end-row_start+1)-(HEIGHT-row_start)), MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
					}
					else{ // don't have to wrap around
						MPI_Recv(grid+WIDTH*row_start, WIDTH*(row_end-row_start+1), MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
					}
					//printf("Rank %d receives rows %d to %d from %d at t=%d\n", my_rank, row_start, row_end, prev_rank, t);
				}

				row_start = ((my_t+1)*ROWS_PER_GPU+(gen-1)) % HEIGHT;
				row_end = row_start+ROWS_PER_GPU-1;
				// may not need to calculate all ROWS_PER_GPU at the last iteration
				// if HEIGHT is not divisible by ROWS_PER_GPU
				if (my_t+1 == iters-1 && HEIGHT % ROWS_PER_GPU != 0)
					row_end = row_start+(HEIGHT % ROWS_PER_GPU)-1;
				
				if (row_end >= HEIGHT){ //rows wrap around so it was sent in 2 chunks
					MPI_Recv(grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL,prev_rank, 0, MPI_COMM_WORLD, status);
					MPI_Recv(grid, WIDTH*((row_end-row_start+1)-(HEIGHT-row_start)), MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
				}
				else{ // don't have to wrap around
					MPI_Recv(grid+WIDTH*row_start, WIDTH*(row_end-row_start+1), MPI_C_BOOL, prev_rank, 0, MPI_COMM_WORLD, status);
				}
				//printf("Rank %d receives rows %d to %d from %d at t=%d\n", my_rank, row_start, row_end, prev_rank, t);
			}

			// Calcuate my section 
			row_start = (my_t*ROWS_PER_GPU+gen) % HEIGHT;
			row_end = row_start+ROWS_PER_GPU-1;
			if (my_t == iters-1 && HEIGHT % ROWS_PER_GPU != 0)
				row_end = row_start+(HEIGHT % ROWS_PER_GPU)-1;
			//printf("Rank %d processes rows %d to %d at t=%d for gen %d\n", my_rank, row_start, row_end, t, gen);
			run_kernel_section(grid, next_grid, WIDTH, HEIGHT, row_start, row_end);
			
			// Send data to the next rank
			if (gen+1 < num_steps){
				//printf("Rank %d sends rows %d to %d to rank %d at t=%d\n",  my_rank, row_start, row_end, next_rank, t);
				if (row_end >= HEIGHT){ //rows wrap around so must send in 2 chunks
					MPI_Isend(next_grid+WIDTH*row_start, WIDTH*(HEIGHT-row_start), MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, &req);
					MPI_Isend(next_grid, WIDTH*((row_end-row_start+1)-(HEIGHT-row_start)), MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, &req);
				}
				else{ // don't have to wrap around
					MPI_Isend(next_grid+WIDTH*row_start, WIDTH*(row_end-row_start+1), MPI_C_BOOL, next_rank, 0, MPI_COMM_WORLD, &req);
				}
			}
			my_t++;
		} 
		// if we've finished calculating this generation
		else if (my_t == iters){ 
			gen += num_ranks; //reuse this rank in num_ranks generations from now
			if (gen < num_steps){
				my_t = 0; //restarting time when we calculate the next gen
				start_t = (gen/num_ranks)*wasted_time + 2*gen; //start time for the next generation			
			}
		}
	}
	end_cycles = clock_now();
	//printf("Rank %d finished in %lf\n",  my_rank, ((double)(end_cycles-start_cycles))/clock_frequency);

	MPI_Barrier(MPI_COMM_WORLD);
	if (my_rank == (num_steps-1) % num_ranks){
		printf("Results:\n");
		printf("- %lu simulations\n", num_steps);
		printf("- grid size %lu\n", WIDTH);
		printf("- %d ranks\n", num_ranks);
		printf("- pipeline strategy\n");
		printf("- time taken to run: %lf\n", ((double)(end_cycles-start_cycles))/clock_frequency);
	}

	// Print output
	/*if (my_rank == (num_steps-1) % num_ranks){
		printf("\nRank %d output (%lu generations have passed)\n", my_rank, num_steps);
		for(int r = 0; r < HEIGHT; r++){
			for(int c = 0; c < WIDTH; c++){
				printf("%d ", next_grid[r*WIDTH + c]);
			}
			printf("\n");
		}	
	}*/

	free_cudamemory(grid, next_grid);
}

//for testing purposes 
//load hardcode config
void load_hardcode_config(const char* hc_config, bool* grid, unsigned long width, unsigned long height){
	printf("%lu %lu %s\n", width, height, hc_config);

	memset(grid, 0, width * height * sizeof(bool));

	if ( strncmp(hc_config, "acorn", 5)==0){
		for(int r = 0; r < 4; r++){
			for(int c = 0; c < 7; c++){
				grid[r*width + c] = ACORN[r][c];
			}
		}
	} else if (strncmp(hc_config, "beacon", 6)==0){
		for(int r = 0; r < 4; r++){
			for(int c = 0; c < 4; c++){
				grid[r*width + c] = BEACON[r][c];
			}
		}
	} else if(strncmp(hc_config, "beehive", 7)==0){
		for(int r = 0; r < 3; r++){
			for(int c = 0; c < 4; c++){
				grid[r*width + c] = BEEHIVE[r][c];
			}
		}
	} else if (strncmp(hc_config, "glider", 6)==0){
		for(int r = 0; r < 3; r++){
			for(int c = 0; c < 3; c++){
				grid[r*width + c] = GLIDER[r][c];
			}
		}
	} else if (strncmp(hc_config, "traffic-light", 13)==0){
		for(int r = 0; r < 2; r++){
			for(int c = 0; c < 3; c++){
				grid[r*width + c] = TRAFFIC_LIGHT[r][c];
			}
		}
	} 

	/*printf("INPUT:\n");
	for(int r = 0; r < height; r++){
		for(int c = 0; c < width; c++){
			printf("%d ", grid[r*width + c]);
		}
		printf("\n");
	}	*/

}

// for testing purposes
// load input file
void load_input_file(char* filename, bool* grid,  unsigned long HEIGHT,  unsigned long WIDTH){
	// since each rank computes a separate generation,
	// only rank 0 needs the initial grid

	FILE *fp = fopen(filename, "r");
	char* line = (char*) malloc(WIDTH*sizeof(char));
	int grid_index = 0;
	while(fgets(line, WIDTH*HEIGHT, fp)){
		//copy buffer contents to grid
		for (int i = 0; i < WIDTH; i++){
			if (line[i] == '*'){
				grid[grid_index] = false;
			} else if (line[i] == '#'){
				grid[grid_index] = true;
			}
			grid_index++;
		}
	}
	free(line);
	fclose(fp);

	//test print
	printf("INPUT:\n");
	for(int r = 0; r < HEIGHT; r++){
		for(int c = 0; c < WIDTH; c++){
			printf("%d ", grid[r*WIDTH + c]);
		}
		printf("\n");
	}	
}