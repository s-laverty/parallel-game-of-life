#ifndef PIPELINED_H
#define PIPELINED_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>
#include <math.h>

#include "clockcycle.h"

// these have to match the WIDTH & HEIGHT in cuda-kernels.cu
#define WIDTH 20
#define HEIGHT 11
#define ROWS_PER_GPU 4 //number of rows computed by each GPU every time step
#define clock_frequency 512000000


void run_pipelined(int my_rank, unsigned long num_steps);

#endif // PIPELINED_H