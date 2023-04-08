/**
 * @file simulation.c
 * @author Steven Laverty (lavers@rpi.edu)
 * @brief This file defines the procedure for parallelized game of life simulation
 * intercommunication.
 * @version 0.1
 * @date 2023-04-07
 */

#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include "clockcycle.h"
#include "grid.h"

#define CELL_EXCHANGE_TAG 0

/** The communicator and ranks of neighboring grid views using a horizontal striped layout. */
typedef struct
{
    /** The communicator shared with neighbors. */
    MPI_Comm comm;
    /** The rank of the view above. */
    int above;
    /** The rank of the view below. */
    int below;
} GridViewNeighborsStriped;

/** The communicator and ranks of neighboring grid views using a brick-style layout. */
typedef struct
{
    /** The communicator shared with neighbors. */
    MPI_Comm comm;
    /** The rank of the view to the upper-left. */
    int above_left;
    /** The rank of the view to the upper-right. */
    int above_right;
    /** The alignment of the view split above the view. */
    size_t above_align;
    /** The rank of the view to the left. */
    int left;
    /** The rank of the view to the right. */
    int right;
    /** The rank of the view to the lower-left. */
    int below_left;
    /** The rank of the view to the lower-right. */
    int below_right;
    /** The alignment of the view split below the view. */
    size_t below_align;
} GridViewNeighborsBrick;

/**
 * @brief Exchange information about border cells with all neighboring views using a horizontal
 * striped layout.
 *
 * @param view A view of the global data grid.
 * @param neighbors All neighboring views.
 */
void exchange_border_cells_striped(const GridView *view, const GridViewNeighborsStriped *neighbors)
{
    // Define MPI send/recv requests.
    MPI_Request recv_requests[2];
    MPI_Request send_requests[2];

    // Make receive requests.
    MPI_Irecv(view->grid[0] + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->above,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 0);
    MPI_Irecv(view->grid[view->height + 1] + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->below,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 1);

    // Make send requests.
    MPI_Isend(view->grid[1] + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->above,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 0);
    MPI_Isend(view->grid[view->height] + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->below,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 1);

    // Copy left-right borders to padding.
    for (size_t i = 1; i <= view->height; i++)
    {
        view->grid[i][0] = view->grid[i][view->width];
        view->grid[i][view->width + 1] = view->grid[i][1];
    }

    // Wait for recv requests to complete.
    MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);

    // Copy padding to corners.
    view->grid[0][0] = view->grid[0][view->width];
    view->grid[0][view->width + 1] = view->grid[0][1];
    view->grid[view->height + 1][0] = view->grid[view->height + 1][view->width];
    view->grid[view->height + 1][view->width + 1] = view->grid[view->height + 1][1];

    // Wait for send requests to complete.
    MPI_Waitall(2, send_requests, MPI_STATUSES_IGNORE);
}

/**
 * @brief Exchange information about border cells with all neighboring views using a brick-style
 * layout.
 *
 * @param view A view of the global data grid.
 * @param neighbors All neighboring views.
 */
void exchange_border_cells_brick(const GridView *view, const GridViewNeighborsBrick *neighbors)
{
    // Define send/recv buffers and MPI send/recv requests.
    bool *left_recv_buf = NULL, *left_send_buf = NULL;
    bool *right_recv_buf = NULL, *right_send_buf = NULL;
    MPI_Request recv_requests[6];
    MPI_Request send_requests[6];

    // Allocate and initialize send/recv buffers.
    if (neighbors->left != MPI_PROC_NULL)
    {
        left_recv_buf = (bool *)malloc(view->height * sizeof(bool));
        left_send_buf = (bool *)malloc(view->height * sizeof(bool));
        get_col(left_send_buf, view->grid, 1, 1, view->height);
    }
    if (neighbors->right != MPI_PROC_NULL)
    {
        right_recv_buf = (bool *)malloc(view->height * sizeof(bool));
        right_send_buf = (bool *)malloc(view->height * sizeof(bool));
        get_col(right_send_buf, view->grid, view->width, 1, view->height);
    }

    // Make receive requests.
    MPI_Irecv(view->grid[0],
              neighbors->above_align + 1,
              MPI_C_BOOL,
              neighbors->above_left,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 0);
    MPI_Irecv(view->grid[0] + neighbors->above_align + 1,
              view->width + 1 - neighbors->above_align,
              MPI_C_BOOL,
              neighbors->above_right,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 1);
    MPI_Irecv(left_recv_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->left,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 2);
    MPI_Irecv(right_recv_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->right,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 3);
    MPI_Irecv(view->grid[view->height + 1],
              neighbors->below_align + 1,
              MPI_C_BOOL,
              neighbors->below_left,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 4);
    MPI_Irecv(view->grid[view->height + 1] + neighbors->below_align + 1,
              view->width + 1 - neighbors->below_align,
              MPI_C_BOOL,
              neighbors->below_right,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 5);

    // Make send requests.
    MPI_Isend(view->grid[1] + 1,
              neighbors->above_align + 1,
              MPI_C_BOOL,
              neighbors->above_left,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 0);
    MPI_Isend(view->grid[1] + neighbors->above_align,
              view->width + 1 - neighbors->above_align,
              MPI_C_BOOL,
              neighbors->above_right,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 1);
    MPI_Isend(left_send_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->left,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 2);
    MPI_Isend(right_send_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->right,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 3);
    MPI_Isend(view->grid[view->height] + 1,
              neighbors->below_align + 1,
              MPI_C_BOOL,
              neighbors->below_left,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 4);
    MPI_Isend(view->grid[view->height] + neighbors->below_align,
              view->width + 1 - neighbors->below_align,
              MPI_C_BOOL,
              neighbors->below_right,
              CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 5);

    // Wait for recv requests to complete.
    MPI_Waitall(6, recv_requests, MPI_STATUSES_IGNORE);

    // Copy from and free recv buffers.
    if (neighbors->left != MPI_PROC_NULL)
    {
        set_col(view->grid, left_recv_buf, 0, 1, view->height);
        free(left_recv_buf);
    }
    if (neighbors->right != MPI_PROC_NULL)
    {
        set_col(view->grid, left_recv_buf, view->width + 1, 1, view->height);
        free(right_recv_buf);
    }

    // Wait for send requests to complete.
    MPI_Waitall(6, send_requests, MPI_STATUSES_IGNORE);

    // Free send buffers.
    if (neighbors->left != MPI_PROC_NULL)
        free(left_send_buf);
    if (neighbors->right != MPI_PROC_NULL)
        free(right_send_buf);
}
