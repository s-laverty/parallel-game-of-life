#ifndef IO_H
#define IO_H
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "grid.h"
#include "clockcycle.h"

#define FILENAME "temp.txt"

void saveToFile(char* filename, int* array, int width, int height, MPI_Comm comm);
int* loadFromFile(char* filename, int* height, int* width, MPI_Comm comm);


#endif //IO_H
