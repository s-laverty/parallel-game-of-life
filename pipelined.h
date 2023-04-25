#ifndef PIPELINED_H
#define PIPELINED_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <math.h>
#include <limits.h>
#include "grid.h"

#include "clockcycle.h"


#define INPUT_FILE "test_board_3.txt"
#define ROWS_PER_GPU 10 //number of rows computed by each GPU every time step
#define clock_frequency 512000000


void run_pipelined(int my_rank, unsigned long HEIGHT, unsigned long WIDTH, unsigned long num_steps, const char* hc_config);

#endif // PIPELINED_H