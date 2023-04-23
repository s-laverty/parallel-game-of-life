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
 * @param fh The MPI file.
 */
int* loadFromFile(char* filename, int num_elements, MPI_Comm comm, MPI_File fh){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int elements_per_rank = num_elements/size;
    int* buffer = (int*) malloc(elements_per_rank * sizeof(int));

    MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset offset = (MPI_Offset)(rank * elements_per_rank + (rank < remainder ? rank : remainder));
    MPI_File_set_view(fh, offset, MPI_INT, MPI_INT, "native", MPI_INFO_NULL);

    MPI_File_read(fh, buf, num_rows * num_cols, MPI_INT, &status);

    MPI_File_close(&fh);

    return buffer;
}

/**
 * @brief Saves an array into a file. Format is first line has the width then height. Then the saved data.
 *
 * @param filename The name of the file to save into.
 * @param array The array of ints that will be saved.
 * @param num_elements Size of the total array.
 * @param comm The MPI environment.
 */
void saveToFile(char* filename, int* array, int num_elements, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int elements_per_rank = num_elements / size;
    int remainder = num_elements % size;


    // Open the file for writing
    MPI_File file;
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    /*/ Write the width and height to the first line of the file
     * Commented out as the file format most likely doesn't require this
    if (rank == 0) {
        char dimensions[256];
        snprintf(dimensions, 256, "%d %d\n", width, height);
        MPI_File_write(file, dimensions, strlen(dimensions), MPI_CHAR, MPI_STATUS_IGNORE);
    }*/

    // Write the array to the file
    MPI_Offset offset = (MPI_Offset)(rank * elements_per_rank + (rank < remainder ? rank : remainder));
    MPI_File_seek(file, offset * sizeof(int) + sizeof(char) * 256, MPI_SEEK_SET);
    MPI_File_write(file, array, num_elements, MPI_INT, MPI_STATUS_IGNORE);

    // Close the file and free memory
    MPI_File_close(&file);
}
