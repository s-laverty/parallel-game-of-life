/**
 * @file io.c
 * @author Albert Drewke (drewka@rpi.edu)
 * @brief This file contains the Parallel IO for the simulation
 * @version 0.1
 * @date 2023-04-21
 */

#include "io.h"

/**
 * @brief Loads in a int array from a filename and gives the height and width for the array. Format is first line has the width then height. Then the saved data.
 *
 * @param filename The name of the file to read from.
 * @param height Int pointer to receive height value.
 * @param width Int pointer to receive width value.
 * @param comm The current MPI environment.
 */
int* loadFromFile(char* filename, int* height, int* width, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int* global_array = NULL;

    // Read dimensions from file on rank 0
    if (rank == 0) {
        FILE* file = fopen(filename, "r");
        if (file == NULL) {
            fprintf(stderr, "Could not open file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(file, "%d %d", width, height);
        fclose(file);
        global_array = (int*) malloc(*height * *width * sizeof(int));
    }

    // Broadcast dimensions to all processes
    MPI_Bcast(height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(width, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide the work among processes
    int rows_per_process = *height / size;
    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    if (rank == size - 1) {
        end_row = *height;
    }

    // Allocate memory for local array
    int local_height = end_row - start_row;
    int* local_array = (int*) malloc(local_height * *width * sizeof(int));

    // Read data from file into local array
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Could not open file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return NULL;
    }
    // Skip first line
    fscanf(file, "%*[^\n]\n");
    // Skip lines before start_row
    for (int i = 0; i < start_row; i++) {
        fscanf(file, "%*[^\n]\n");
    }
    // Read lines into local array
    for (int i = 0; i < local_height; i++) {
        for (int j = 0; j < *width; j++) {
            fscanf(file, "%d", &local_array[i**width+j]);
        }
        fscanf(file, "\n");
    }
    fclose(file);

    // Gather local arrays into global array on rank 0
    int* recv_counts = (int*) malloc(size * sizeof(int));
    int* displacements = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        recv_counts[i] = (*height / size) * *width;
        displacements[i] = i * recv_counts[i];
    }
    MPI_Gatherv(local_array, local_height * *width, MPI_INT,
                global_array, recv_counts, displacements, MPI_INT,
                0, MPI_COMM_WORLD);

    // Free memory
    free(local_array);
    free(recv_counts);
    free(displacements);

    return global_array;
}

/**
 * @brief Saves an array into a file. Format is first line has the width then height. Then the saved data.
 *
 * @param filename The name of the file to save into.
 * @param array The array of ints that will be saved.
 * @param width The width of the array.
 * @param height The height of the array.
 * @param comm The MPI environment.
 */
void saveToFile(char* filename, int* array, int width, int height, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int num_elements = width * height;
    int elements_per_rank = num_elements / size;
    int remainder = num_elements % size;

    // Determine the number of elements each rank will write to the file
    int my_elements = elements_per_rank;
    if (rank < remainder) {
        my_elements++;
    }

    // Allocate memory for the portion of the array this rank will write
    int* my_array = (int*)malloc(my_elements * sizeof(int));
    int i, k = 0;
    for (i = rank; i < num_elements; i += size) {
        my_array[k++] = array[i];
    }

    // Open the file for writing
    MPI_File file;
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // Write the width and height to the first line of the file
    if (rank == 0) {
        char dimensions[256];
        snprintf(dimensions, 256, "%d %d\n", width, height);
        MPI_File_write(file, dimensions, strlen(dimensions), MPI_CHAR, MPI_STATUS_IGNORE);
    }

    // Write the array to the file
    MPI_Offset offset = (MPI_Offset)(rank * elements_per_rank + (rank < remainder ? rank : remainder));
    MPI_File_seek(file, offset * sizeof(int) + sizeof(char) * 256, MPI_SEEK_SET);
    MPI_File_write(file, my_array, my_elements, MPI_INT, MPI_STATUS_IGNORE);

    // Close the file and free memory
    MPI_File_close(&file);
    free(my_array);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    char* filename = FILENAME;
    int width = 7;
    int height = 3;
    int array[3][7] = {
            {0, 1, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0, 0},
            {1, 1, 0, 0, 1, 1, 1},
    };

    saveToFile(filename, array, width, height, MPI_COMM_WORLD);

    MPI_Finalize();
    printf("hi :)\n");
    return 0;
}