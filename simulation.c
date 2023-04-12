/**
 * @file simulation.c
 * @author Steven Laverty (lavers@rpi.edu)
 * @brief This file defines the procedure for parallelized game of life simulation
 * intercommunication.
 * @version 0.1
 * @date 2023-04-07
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "clockcycle.h"
#include "grid.h"

/** Border cell exchange tags. Relative position of receiver. */
#define BORDER_CELL_EXCHANGE_TAG 0

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
    MPI_Irecv(row_ptr(&view->grid, 0) + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->above,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 0);
    MPI_Irecv(row_ptr(&view->grid, view->height + 1) + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->below,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 1);

    // Make send requests.
    MPI_Isend(row_ptr(&view->grid, 1) + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->above,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 0);
    MPI_Isend(row_ptr(&view->grid, view->height) + 1,
              view->width,
              MPI_C_BOOL,
              neighbors->below,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 1);

#if WRAP_GLOBAL_GRID
    {
        // Copy left-right borders to padding.
        bool *row = row_ptr(&view->grid, 1);
        for (size_t i = 0; i < view->height; i++, row += view->grid.width)
        {
            row[0] = row[view->width];
            row[view->width + 1] = row[1];
        }
    }
#endif

    // Wait for recv requests to complete.
    MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);

#if WRAP_GLOBAL_GRID
    {
        // Copy padding to corners.
        bool *row = row_ptr(&view->grid, 0);
        row[0] = row[view->width];
        row[view->width + 1] = row[1];
        row = row_ptr(&view->grid, view->height + 1);
        row[0] = row[view->width];
        row[view->width + 1] = row[1];
    }
#endif

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
        get_col(left_send_buf, &view->grid, 1, 1, view->height);
    }
    if (neighbors->right != MPI_PROC_NULL)
    {
        right_recv_buf = (bool *)malloc(view->height * sizeof(bool));
        right_send_buf = (bool *)malloc(view->height * sizeof(bool));
        get_col(right_send_buf, &view->grid, view->width, 1, view->height);
    }

    // Make receive requests.
    MPI_Irecv(row_ptr(&view->grid, 0),
              neighbors->above_align + 1,
              MPI_C_BOOL,
              neighbors->above_left,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 0);
    MPI_Irecv(row_ptr(&view->grid, 0) + neighbors->above_align + 1,
              view->width + 1 - neighbors->above_align,
              MPI_C_BOOL,
              neighbors->above_right,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 1);
    MPI_Irecv(left_recv_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->left,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 2);
    MPI_Irecv(right_recv_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->right,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 3);
    MPI_Irecv(row_ptr(&view->grid, view->height + 1),
              neighbors->below_align + 1,
              MPI_C_BOOL,
              neighbors->below_left,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 4);
    MPI_Irecv(row_ptr(&view->grid, view->height + 1) + neighbors->below_align + 1,
              view->width + 1 - neighbors->below_align,
              MPI_C_BOOL,
              neighbors->below_right,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              recv_requests + 5);

    // Make send requests.
    MPI_Isend(row_ptr(&view->grid, 1) + 1,
              neighbors->above_align + 1,
              MPI_C_BOOL,
              neighbors->above_left,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 0);
    MPI_Isend(row_ptr(&view->grid, 1) + neighbors->above_align,
              view->width + 1 - neighbors->above_align,
              MPI_C_BOOL,
              neighbors->above_right,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 1);
    MPI_Isend(left_send_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->left,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 2);
    MPI_Isend(right_send_buf,
              view->height,
              MPI_C_BOOL,
              neighbors->right,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 3);
    MPI_Isend(row_ptr(&view->grid, view->height) + 1,
              neighbors->below_align + 1,
              MPI_C_BOOL,
              neighbors->below_left,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 4);
    MPI_Isend(row_ptr(&view->grid, view->height) + neighbors->below_align,
              view->width + 1 - neighbors->below_align,
              MPI_C_BOOL,
              neighbors->below_right,
              BORDER_CELL_EXCHANGE_TAG,
              neighbors->comm,
              send_requests + 5);

    // Wait for recv requests to complete.
    MPI_Waitall(6, recv_requests, MPI_STATUSES_IGNORE);

    // Copy from and free recv buffers.
    if (neighbors->left != MPI_PROC_NULL)
    {
        set_col(&view->grid, left_recv_buf, 0, 1, view->height);
        free(left_recv_buf);
    }
    if (neighbors->right != MPI_PROC_NULL)
    {
        set_col(&view->grid, left_recv_buf, view->width + 1, 1, view->height);
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

/**
 * @brief Initialize the grid member of an otherwise-initialized GridView object.
 *
 * @param view The partially-initialized GridView object.
 */
static void _alloc_grid_view(GridView *view)
{
    view->grid.width = view->width + 2;
    view->grid.height = view->height + 2;
    view->grid.data = (bool *)calloc(view->grid.width * view->grid.height, sizeof(bool));
}

/**
 * @brief Get the global grid view and neighbors for this rank using the vertical striped
 * fragmentation strategy. Requires at least 2 ranks.
 *
 * @param view The global grid view object to initialize.
 * @param neighbors The neighbors object to initialize. neighbors->comm should already be
 * initialized.
 * @param num_rows The number of rows in the global data grid.
 * @param num_cols The number of columns in the global data grid.
 * @return true upon success.
 * @return false upon failure; i.e. not enough ranks.
 */
bool get_view_striped(GridView *view,
                      GridViewNeighborsStriped *neighbors,
                      size_t num_rows,
                      size_t num_cols)
{
    // Validate communicator
    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(neighbors->comm, &world_size);
    MPI_Comm_rank(neighbors->comm, &world_rank);
    if (world_size < 2)
        return false;

    // Determine stripe layout
    div_t height = div(num_rows, world_size);

    // Initialize view
    bool height_pad = world_rank < height.rem;
    view->row_start = world_rank * height.quot + (height_pad ? world_rank : height.rem);
    view->height = height.quot + height_pad;
    view->col_start = 0;
    view->width = num_cols;
    _alloc_grid_view(view);

    // Initialize neighbors
#if WRAP_GLOBAL_GRID
    neighbors->above = (world_rank > 0 ? world_rank : world_size) - 1;
    neighbors->below = (world_rank + 1) % world_size;
#else
    neighbors->above = world_rank > 0 ? world_rank - 1 : MPI_PROC_NULL;
    neighbors->below = world_rank + 1 < world_size ? world_rank + 1 : MPI_PROC_NULL;
#endif

    return true;
}

/**
 * @brief Get the global grid view and neighbors for this rank using the brick-style
 * fragmentation strategy. Requires at least 4 ranks, total number must be divisible by 2.
 *
 * @param view The global grid view object to initialize.
 * @param neighbors The neighbors object to initialize. neighbors->comm should already be
 * initialized.
 * @param num_rows The number of rows in the global data grid.
 * @param num_cols The number of columns in the global data grid.
 * @return true upon success.
 * @return false upon failure; i.e. not enough ranks.
 */
bool get_view_brick(GridView *view,
                    GridViewNeighborsBrick *neighbors,
                    size_t num_rows,
                    size_t num_cols)
{
    // Validate communicator
    int world_size = 0;
    int world_rank = 0;
    MPI_Comm_size(neighbors->comm, &world_size);
    MPI_Comm_rank(neighbors->comm, &world_rank);
    if (world_size < 4)
        return false;
    if (world_size % 2)
        return false;

    // Determine brick layout
    // Prefer using larger factor as number of rows.
    size_t num_brick_cols = (size_t)sqrt((double)world_size);
    while (world_size % num_brick_cols)
        num_brick_cols--;
    size_t num_brick_rows = world_size / num_brick_cols;
    if (num_brick_rows % 2)
    {
        // Number of rows MUST be even; swap rows / cols
        size_t tmp = num_brick_rows;
        num_brick_rows = num_brick_cols;
        num_brick_cols = tmp;
    }
    div_t height = div(num_rows, num_brick_rows);
#if WRAP_GLOBAL_GRID
    div_t width = div(num_cols, num_brick_cols);
    size_t offset = (width.quot + 1) / 2;
#else
    div_t width = div(num_cols, num_brick_cols * 2 - 1);
    size_t offset = width.quot;
    width.quot *= 2;
    if (width.rem >= num_brick_cols)
    {
        width.quot += 1;
        offset += 1;
        width.rem -= num_brick_cols;
    }
#endif
    if (width.quot < 2)
        return false;

    // Initialize view
    div_t brick_idx = div(world_rank, num_brick_cols); // quot = row_idx; rem = col_idx
    bool is_odd_row = brick_idx.quot % 2;
    bool height_pad = brick_idx.quot < height.rem;
    view->row_start = brick_idx.quot * height.quot + (height_pad ? brick_idx.quot : height.rem);
    view->height = height.quot + height_pad;
#if WRAP_GLOBAL_GRID
    bool width_pad = brick_idx.rem < width.rem;
    view->col_start = brick_idx.rem * width.quot + (width_pad ? brick_idx.rem : width.rem) +
                      is_odd_row * offset; // offset every other row
    view->width = width.quot + width_pad;
#else
    if (is_odd_row && !brick_idx.rem)
    {
        view->col_start = 0;
        view->width = offset;
    }
    else
    {
        size_t col_idx = brick_idx.rem - is_odd_row;
        bool width_pad = col_idx < width.rem;
        view->col_start = col_idx * width.quot + (width_pad ? col_idx : width.rem) +
                          is_odd_row * offset; // offset every other row
        view->width = ((col_idx + 1 < num_cols) ? width.quot : offset) + width_pad;
    }
#endif
    _alloc_grid_view(view);

    // Initialize neighbors
    neighbors->above_align = neighbors->below_align = is_odd_row ? view->width - offset : offset;
#if WRAP_GLOBAL_GRID
    neighbors->left = world_rank - brick_idx.rem +
                      (brick_idx.rem + num_brick_cols - 1) % num_brick_cols;
    neighbors->right = world_rank - brick_idx.rem +
                       (brick_idx.rem + 1) % num_brick_cols;
    neighbors->above_left = neighbors->above_right = (world_rank + world_size - num_brick_cols) %
                                                     world_size;
    neighbors->below_left = neighbors->below_right = (world_rank + num_brick_cols) % world_size;
    if (is_odd_row)
    {
        neighbors->above_right += neighbors->right - world_rank;
        neighbors->above_right += neighbors->right - world_rank;
    }
    else
    {
        neighbors->above_left += neighbors->left - world_rank;
        neighbors->below_left += neighbors->left - world_rank;
    }
#else
    neighbors->left = (brick_idx.rem > 0) ? world_rank - 1 : MPI_PROC_NULL;
    neighbors->right = (brick_idx.rem + 1 < num_brick_cols) ? world_rank + 1 : MPI_PROC_NULL;
    neighbors->above_left = neighbors->above_right = (brick_idx.quot > 0)
                                                         ? world_rank - num_brick_cols
                                                         : MPI_PROC_NULL;
    neighbors->below_left = neighbors->below_right = (brick_idx.quot + 1 < num_brick_rows)
                                                         ? world_rank + num_brick_cols
                                                         : MPI_PROC_NULL;
    if (is_odd_row)
    {
        neighbors->above_left = (neighbors->above_left != MPI_PROC_NULL &&
                                 neighbors->right != MPI_PROC_NULL)
                                    ? neighbors->above_left - 1
                                    : MPI_PROC_NULL;
        neighbors->below_left = (neighbors->below_left != MPI_PROC_NULL &&
                                  neighbors->right != MPI_PROC_NULL)
                                     ? neighbors->below_left - 1
                                     : MPI_PROC_NULL;
    }
    else
    {
        neighbors->above_right = (neighbors->above_right != MPI_PROC_NULL &&
                                  neighbors->right != MPI_PROC_NULL)
                                     ? neighbors->above_right + 1
                                     : MPI_PROC_NULL;
        neighbors->below_right = (neighbors->below_right != MPI_PROC_NULL &&
                                  neighbors->right != MPI_PROC_NULL)
                                     ? neighbors->below_right + 1
                                     : MPI_PROC_NULL;
    }
#endif

    return true;
}

int main(int argc, char const *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
