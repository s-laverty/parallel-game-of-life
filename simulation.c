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
#include "pipelined.h"

// CUDA functions
extern void cuda_init_gridview(GridView *grid, int my_rank);
extern void run_kernel_nowrap(GridView *grid);
extern void free_cudamem_gridview(GridView *grid);

#define MAX_RANKS 96

#ifndef DEBUG

#define MPI_CELL_Datatype MPI_C_BOOL

#else

#define MPI_CELL_Datatype MPI_CHAR

#endif

// Hardcoded initial grid configurations
Cell_t const ACORN[3][7] = {
    {0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0, 0},
    {1, 1, 0, 0, 1, 1, 1},
};
Cell_t const BEACON[4][4] = {
    {1, 1, 0, 0},
    {1, 0, 0, 0},
    {0, 0, 0, 1},
    {0, 0, 1, 1},
};
Cell_t const BEEHIVE[3][4] = {
    {0, 1, 1, 0},
    {1, 0, 0, 1},
    {0, 1, 1, 0},
};
Cell_t const GLIDER[3][3] = {
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 1},
};
Cell_t const TRAFFIC_LIGHT[2][3] = {
    {0, 1, 0},
    {1, 1, 1},
};

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
void exchange_border_cells_striped(GridView *view, GridViewNeighborsStriped const *neighbors)
{
    /** Border cell exchange tags */
    enum
    {
        TO_ABOVE,
        TO_BELOW
    };

    // Define MPI send/recv requests.
    MPI_Request recv_requests[2];
    MPI_Request send_requests[2];

    // Make receive requests.
    MPI_Irecv(row_ptr(&view->grid, 0) + 1,
              view->width,
              MPI_CELL_Datatype,
              neighbors->above,
              TO_BELOW,
              neighbors->comm,
              recv_requests + 0);
    MPI_Irecv(row_ptr(&view->grid, view->height + 1) + 1,
              view->width,
              MPI_CELL_Datatype,
              neighbors->below,
              TO_ABOVE,
              neighbors->comm,
              recv_requests + 1);

    // Make send requests.
    MPI_Isend(row_ptr(&view->grid, 1) + 1,
              view->width,
              MPI_CELL_Datatype,
              neighbors->above,
              TO_ABOVE,
              neighbors->comm,
              send_requests + 0);
    MPI_Isend(row_ptr(&view->grid, view->height) + 1,
              view->width,
              MPI_CELL_Datatype,
              neighbors->below,
              TO_BELOW,
              neighbors->comm,
              send_requests + 1);

#if WRAP_GLOBAL_GRID
    {
        // Copy left-right borders to padding.
        Cell_t *row = row_ptr(&view->grid, 1);
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
        Cell_t *row = row_ptr(&view->grid, 0);
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
void exchange_border_cells_brick(GridView *view, GridViewNeighborsBrick const *neighbors)
{
    /** Border cell exchange tags */
    enum
    {
        TO_ABOVE_LEFT,
        TO_ABOVE_RIGHT,
        TO_LEFT,
        TO_RIGHT,
        TO_BELOW_LEFT,
        TO_BELOW_RIGHT
    };

    // Define send/recv buffers and MPI send/recv requests. The buffers may be effectively unused,
    // but they still must point to real arrays (MPI_IRecv input validation).
    Cell_t *left_recv_buf = view->grid.data, *left_send_buf = view->grid.data;
    Cell_t *right_recv_buf = view->grid.data, *right_send_buf = view->grid.data;
    MPI_Request recv_requests[6];
    MPI_Request send_requests[6];

    // Allocate and initialize send/recv buffers.
    if (neighbors->left != MPI_PROC_NULL)
    {
        left_recv_buf = (Cell_t *)malloc(view->height * sizeof(Cell_t));
        left_send_buf = (Cell_t *)malloc(view->height * sizeof(Cell_t));
        get_col(left_send_buf, &view->grid, 1, 1, view->height);
    }
    if (neighbors->right != MPI_PROC_NULL)
    {
        right_recv_buf = (Cell_t *)malloc(view->height * sizeof(Cell_t));
        right_send_buf = (Cell_t *)malloc(view->height * sizeof(Cell_t));
        get_col(right_send_buf, &view->grid, view->width, 1, view->height);
    }

    // Make receive requests.
    MPI_Irecv(row_ptr(&view->grid, 0),
              neighbors->above_align + 1,
              MPI_CELL_Datatype,
              neighbors->above_left,
              TO_BELOW_RIGHT,
              neighbors->comm,
              recv_requests + 0);
    MPI_Irecv(row_ptr(&view->grid, 0) + neighbors->above_align + 1,
              view->width + 1 - neighbors->above_align,
              MPI_CELL_Datatype,
              neighbors->above_right,
              TO_BELOW_LEFT,
              neighbors->comm,
              recv_requests + 1);
    MPI_Irecv(left_recv_buf,
              view->height,
              MPI_CELL_Datatype,
              neighbors->left,
              TO_RIGHT,
              neighbors->comm,
              recv_requests + 2);
    MPI_Irecv(right_recv_buf,
              view->height,
              MPI_CELL_Datatype,
              neighbors->right,
              TO_LEFT,
              neighbors->comm,
              recv_requests + 3);
    MPI_Irecv(row_ptr(&view->grid, view->height + 1),
              neighbors->below_align + 1,
              MPI_CELL_Datatype,
              neighbors->below_left,
              TO_ABOVE_RIGHT,
              neighbors->comm,
              recv_requests + 4);
    MPI_Irecv(row_ptr(&view->grid, view->height + 1) + neighbors->below_align + 1,
              view->width + 1 - neighbors->below_align,
              MPI_CELL_Datatype,
              neighbors->below_right,
              TO_ABOVE_LEFT,
              neighbors->comm,
              recv_requests + 5);

    // Make send requests.
    MPI_Isend(row_ptr(&view->grid, 1) + 1,
              neighbors->above_align + 1,
              MPI_CELL_Datatype,
              neighbors->above_left,
              TO_ABOVE_LEFT,
              neighbors->comm,
              send_requests + 0);
    MPI_Isend(row_ptr(&view->grid, 1) + neighbors->above_align,
              view->width + 1 - neighbors->above_align,
              MPI_CELL_Datatype,
              neighbors->above_right,
              TO_ABOVE_RIGHT,
              neighbors->comm,
              send_requests + 1);
    MPI_Isend(left_send_buf,
              view->height,
              MPI_CELL_Datatype,
              neighbors->left,
              TO_LEFT,
              neighbors->comm,
              send_requests + 2);
    MPI_Isend(right_send_buf,
              view->height,
              MPI_CELL_Datatype,
              neighbors->right,
              TO_RIGHT,
              neighbors->comm,
              send_requests + 3);
    MPI_Isend(row_ptr(&view->grid, view->height) + 1,
              neighbors->below_align + 1,
              MPI_CELL_Datatype,
              neighbors->below_left,
              TO_BELOW_LEFT,
              neighbors->comm,
              send_requests + 4);
    MPI_Isend(row_ptr(&view->grid, view->height) + neighbors->below_align,
              view->width + 1 - neighbors->below_align,
              MPI_CELL_Datatype,
              neighbors->below_right,
              TO_BELOW_RIGHT,
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
        set_col(&view->grid, right_recv_buf, view->width + 1, 1, view->height);
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
 * @brief Helper function for get_view functions. Set the width and height of the grid members of
 * a grid view.
 *
 * @param view The partially-initialized GridView.
 */
static void _set_grid_dims(GridView *view)
{
    view->grid.height = view->height + 2;
    view->grid.width = view->width + 2;
    view->next_grid.height = view->height + 2;
    view->next_grid.width = view->width + 2;
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
    _set_grid_dims(view);

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
 * @return false upon failure; i.e. invalid number of ranks.
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
        view->width = ((col_idx + 1 < num_brick_cols) ? width.quot : offset) + width_pad;
    }
#endif
    _set_grid_dims(view);

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
        neighbors->below_right += neighbors->right - world_rank;
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
                                 neighbors->left != MPI_PROC_NULL)
                                    ? neighbors->above_left - 1
                                    : MPI_PROC_NULL;
        neighbors->below_left = (neighbors->below_left != MPI_PROC_NULL &&
                                 neighbors->left != MPI_PROC_NULL)
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

#ifdef DEBUG
    fprintf(stdout, "[Rank %d] Left rank: %d\n", world_rank, neighbors->left);
    fprintf(stdout, "[Rank %d] Right rank: %d\n", world_rank, neighbors->right);
    fprintf(stdout, "[Rank %d] Above left rank: %d\n", world_rank, neighbors->above_left);
    fprintf(stdout, "[Rank %d] Above right rank: %d\n", world_rank, neighbors->above_right);
    fprintf(stdout, "[Rank %d] Below left rank: %d\n", world_rank, neighbors->below_left);
    fprintf(stdout, "[Rank %d] Below right rank: %d\n", world_rank, neighbors->below_right);
#endif

    return true;
}

/**
 * @brief Copy a portion of an initialization buffer to the view's data buffer.
 *
 * @param view A view of the global data grid.
 * @param buf An initialization buffer.
 * @param row_start The global row start index of the initialization buffer.
 * @param col_start The global column start index of the initialization buffer.
 * @param height The number of rows of the initialization buffer.
 * @param width The number of columns of the initialization buffer.
 */
void init_view(GridView *view,
               Cell_t const *buf,
               size_t row_start,
               size_t col_start,
               size_t height,
               size_t width)
{
    // Start at higher row start
    size_t row_overlap_start = (view->row_start > row_start) ? view->row_start : row_start;
    // End at lower row end
    size_t row_overlap_end = (view->row_start + view->height < row_start + height)
                                 ? view->row_start + view->height
                                 : row_start + height;
    // Start at higher col start
    size_t col_overlap_start = (view->col_start > col_start) ? view->col_start : col_start;
    // End at lower col end
    size_t col_overlap_end = (view->col_start + view->width < col_start + width)
                                 ? view->col_start + view->width
                                 : col_start + width;
    if (col_overlap_end > col_overlap_start)
    {
        size_t length = col_overlap_end - col_overlap_start;
        for (size_t i = row_overlap_start; i < row_overlap_end; i++)
            set_row(&view->grid,
                    buf + (i - row_start) * width + col_overlap_start - col_start,
                    1 + i - view->row_start,
                    1 + col_overlap_start - view->col_start,
                    length);
    }
}

/**
 * @brief Load a view of the global data grid from a file.
 *
 * @param view The grid view.
 * @param grid_height The height of the global data grid.
 * @param grid_width The width of the global data grid.
 * @param comm The communicator shared with other grid views.
 * @param fname The name of the file to load data from.
 * @return true upon success.
 * @return false upon failure.
 */
bool load_grid_view(GridView *view,
                    size_t grid_height,
                    size_t grid_width,
                    MPI_Comm comm,
                    char const *fname)
{
    // Open file
    MPI_File file;
    if (MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &file))
        return false;
    MPI_File_set_view(file,
                      grid_width * view->row_start + view->col_start,
                      MPI_CELL_Datatype,
                      MPI_CELL_Datatype,
                      "native",
                      MPI_INFO_NULL);
    // Perform as much collective I/O as possible
    size_t comm_height;
    MPI_Allreduce(&view->height, &comm_height, 1, MPI_UNSIGNED_LONG, MPI_MIN, comm);
    if (view->col_start + view->width > grid_width)
    {
        // Wrapping from right to left edge
        size_t right_len = grid_width - view->col_start;
        size_t left_len = view->width - right_len;
        for (size_t row = 0; row < comm_height; row++)
        {
            MPI_File_read_at_all(file,
                                 row * grid_width,
                                 row_ptr(&view->grid, row),
                                 right_len,
                                 MPI_CELL_Datatype,
                                 MPI_STATUS_IGNORE);
            MPI_File_read_at_all(file,
                                 row * grid_width - view->col_start,
                                 row_ptr(&view->grid, row) + right_len,
                                 left_len,
                                 MPI_CELL_Datatype,
                                 MPI_STATUS_IGNORE);
        }
        for (size_t row = comm_height; row < view->height; row++)
        {
            MPI_File_read_at(file,
                             row * grid_width,
                             row_ptr(&view->grid, row),
                             right_len,
                             MPI_CELL_Datatype,
                             MPI_STATUS_IGNORE);
            MPI_File_read_at(file,
                             row * grid_width - view->col_start,
                             row_ptr(&view->grid, row) + right_len,
                             left_len,
                             MPI_CELL_Datatype,
                             MPI_STATUS_IGNORE);
        }
    }
    else
    {
        // No wrapping
        for (size_t row = 0; row < comm_height; row++)
            MPI_File_write_at_all(file,
                                  row * grid_width,
                                  row_ptr(&view->grid, row),
                                  view->width,
                                  MPI_CELL_Datatype,
                                  MPI_STATUS_IGNORE);
        for (size_t row = comm_height; row < view->height; row++)
            MPI_File_write_at(file,
                              row * grid_width,
                              row_ptr(&view->grid, row),
                              view->width,
                              MPI_CELL_Datatype,
                              MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
    return true;
    return true;
}

/**
 * @brief Save a view of the global data grid to a file.
 *
 * @param view The grid view.
 * @param grid_height The height of the global data grid.
 * @param grid_width The width of the global data grid.
 * @param comm The communicator shared with other grid views.
 * @param fname The name of the file to save data to.
 * @return true upon success.
 * @return false upon failure.
 */
bool save_grid_view(GridView const *view,
                    size_t grid_height,
                    size_t grid_width,
                    MPI_Comm comm,
                    char const *fname)
{
    // Open file
    MPI_File file;
    if (MPI_File_open(comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file))
        return false;
    MPI_File_set_view(file,
                      grid_width * view->row_start + view->col_start,
                      MPI_CELL_Datatype,
                      MPI_CELL_Datatype,
                      "native",
                      MPI_INFO_NULL);
    // Perform as much collective I/O as possible
    size_t comm_height;
    MPI_Allreduce(&view->height, &comm_height, 1, MPI_UNSIGNED_LONG, MPI_MIN, comm);
    if (view->col_start + view->width > grid_width)
    {
        // Wrapping from right to left edge
        size_t right_len = grid_width - view->col_start;
        size_t left_len = view->width - right_len;
        for (size_t row = 0; row < comm_height; row++)
        {
            MPI_File_write_at_all(file,
                                  row * grid_width,
                                  row_ptr(&view->grid, row),
                                  right_len,
                                  MPI_CELL_Datatype,
                                  MPI_STATUS_IGNORE);
            MPI_File_write_at_all(file,
                                  row * grid_width - view->col_start,
                                  row_ptr(&view->grid, row) + right_len,
                                  left_len,
                                  MPI_CELL_Datatype,
                                  MPI_STATUS_IGNORE);
        }
        for (size_t row = comm_height; row < view->height; row++)
        {
            MPI_File_write_at(file,
                              row * grid_width,
                              row_ptr(&view->grid, row),
                              right_len,
                              MPI_CELL_Datatype,
                              MPI_STATUS_IGNORE);
            MPI_File_write_at(file,
                              row * grid_width - view->col_start,
                              row_ptr(&view->grid, row) + right_len,
                              left_len,
                              MPI_CELL_Datatype,
                              MPI_STATUS_IGNORE);
        }
    }
    else
    {
        // No wrapping
        for (size_t row = 0; row < comm_height; row++)
            MPI_File_write_at_all(file,
                                  row * grid_width,
                                  row_ptr(&view->grid, row),
                                  view->width,
                                  MPI_CELL_Datatype,
                                  MPI_STATUS_IGNORE);
        for (size_t row = comm_height; row < view->height; row++)
            MPI_File_write_at(file,
                              row * grid_width,
                              row_ptr(&view->grid, row),
                              view->width,
                              MPI_CELL_Datatype,
                              MPI_STATUS_IGNORE);
    }
    MPI_File_close(&file);
    return true;
}

int main(int argc, char *argv[])
{
    /** Fragmentation strategies. */
    enum strategy
    {
        STRAT_STRIPED,
        STRAT_BRICK,
        STRAT_PIPELINE,
        STRAT_MAX
    };
    /** Strategy CLI names. */
    static char const *strategies[] = {
        [STRAT_STRIPED] = "striped",
        [STRAT_BRICK] = "brick",
        [STRAT_PIPELINE] = "pipeline"};
    /** Union type for grid view neighbors. */
    union neighbors
    {
        GridViewNeighborsStriped striped;
        GridViewNeighborsBrick brick;
    };
    /** Hardcode configurations. */
    enum hardcode_config
    {
        HC_CONFIG_ACORN,
        HC_CONFIG_BEACON,
        HC_CONFIG_BEEHIVE,
        HC_CONFIG_GLIDER,
        HC_CONFIG_TRAFFIC_LIGHT,
        HC_CONFIG_MAX,
        HC_CONFIG_NONE,
    };
    /** Hardcode config CLI names. */
    static char const *hardcode_configs[] = {
        [HC_CONFIG_ACORN] = "acorn",
        [HC_CONFIG_BEACON] = "beacon",
        [HC_CONFIG_BEEHIVE] = "beehive",
        [HC_CONFIG_GLIDER] = "glider",
        [HC_CONFIG_TRAFFIC_LIGHT] = "traffic-light",
    };
    /** Grid sizes. */
    enum grid_size
    {
        GRID_SIZE_SMALL,
        GRID_SIZE_MEDIUM,
        GRID_SIZE_LARGE,
        GRID_SIZE_XXL,
        GRID_SIZE_MAX,
    };
    /** Grid size array. */
    static size_t const grid_sizes[] = {
        [GRID_SIZE_SMALL] = 4096,   // 2^12
        [GRID_SIZE_MEDIUM] = 32768, // 2^15
        [GRID_SIZE_LARGE] = 262144, // 2^18
        [GRID_SIZE_XXL] = 1048576,  // 2^20
    };
    /** Grid size CLI names. */
    static char const *grid_size_names[] = {
        [GRID_SIZE_SMALL] = "s",
        [GRID_SIZE_MEDIUM] = "m",
        [GRID_SIZE_LARGE] = "l",
        [GRID_SIZE_XXL] = "xl",
    };
    /** Function pointer to get_view function. */
    typedef bool (*get_view_fn_ptr)(GridView *, union neighbors *, size_t, size_t);
    /** Function pointer to border exchange function. */
    typedef void (*exchange_fn_ptr)(GridView *, union neighbors const *);
    /** Argument parse error string. */
    static char const *arg_parse_err = "Usage: %s [-l checkpoint] [-o checkpoint] [-i hardcode_initializer] [-s grid_size] [-w] strategy num_steps\n";

    MPI_Init(&argc, &argv);

    // Parse input args
    char const *load_checkpoint = NULL;
    char const *save_checkpoint = NULL;
    enum hardcode_config hc_config = HC_CONFIG_NONE;
    enum grid_size grid_size = GRID_SIZE_MEDIUM;
    bool weak_scaling = false;
    int opt;
    while ((opt = getopt(argc, argv, "l:o:i:s:w")) != -1)
    {
        switch (opt)
        {
        case 'l':
            load_checkpoint = optarg;
            break;
        case 'o':
            save_checkpoint = optarg;
            break;
        case 'i':
            hc_config = 0;
            while (strcmp(optarg, hardcode_configs[hc_config]))
                if (++hc_config == HC_CONFIG_MAX)
                {
                    fprintf(stderr,
                            "Hardcode initializer \"%s\" invalid. Must be one of:\n",
                            optarg);
                    for (hc_config = 0; hc_config < HC_CONFIG_MAX; hc_config++)
                        fprintf(stderr, "  %s\n", hardcode_configs[hc_config]);
                    return EXIT_FAILURE;
                }
            break;
        case 's':
            grid_size = 0;
            while (strcmp(optarg, grid_size_names[grid_size]))
                if (++grid_size == GRID_SIZE_MAX)
                {
                    fprintf(stderr,
                            "Grid size \"%s\" invalid. Must be one of:\n",
                            optarg);
                    for (grid_size = 0; grid_size < GRID_SIZE_MAX; grid_size++)
                        fprintf(stderr, "  %s\n", grid_size_names[grid_size]);
                    return EXIT_FAILURE;
                }
            break;
        case 'w':
            weak_scaling = true;
            break;
        default:
            fprintf(stderr, arg_parse_err, argv[0]);
            return EXIT_FAILURE;
        }
    }
#ifdef DEBUG
    printf("Finished processing optional CLI args\n");
#endif
    int pos_argc = argc - optind;
    char **pos_argv = argv + optind;
    if (pos_argc < 2)
    {
        fprintf(stderr, arg_parse_err, argv[0]);
        return EXIT_FAILURE;
    }
    enum strategy strategy = 0;
    while (strcmp(pos_argv[0], strategies[strategy]))
        if (++strategy == STRAT_MAX)
        {
            fprintf(stderr,
                    "Fragmentation strategy \"%s\" invalid. Must be one of:\n",
                    pos_argv[0]);
            for (strategy = 0; strategy < STRAT_MAX; strategy++)
                fprintf(stderr, "  %s\n", strategies[strategy]);
            return EXIT_FAILURE;
        }
    char *endptr;
    unsigned long num_steps = strtoul(pos_argv[1], &endptr, 0);
    if (*endptr)
    {
        fprintf(stderr,
                "%s is an invalid number steps to run the simulation. Must be a positive integer.\n",
                pos_argv[1]);
        return EXIT_FAILURE;
    }
#ifdef DEBUG
    printf("Finished processing positional CLI args\n");
#endif

    // Initialize
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    GridView view;
    union neighbors neighbors;
    get_view_fn_ptr get_view_fn;
    exchange_fn_ptr exchange_fn;
    switch (strategy)
    {
    case STRAT_STRIPED:
        neighbors.striped.comm = MPI_COMM_WORLD;
        get_view_fn = (get_view_fn_ptr)get_view_striped;
        exchange_fn = (exchange_fn_ptr)exchange_border_cells_striped;
        break;
    case STRAT_BRICK:
        neighbors.brick.comm = MPI_COMM_WORLD;
        get_view_fn = (get_view_fn_ptr)get_view_brick;
        exchange_fn = (exchange_fn_ptr)exchange_border_cells_brick;
        break;
    case STRAT_PIPELINE:
        run_pipelined(world_rank, grid_sizes[grid_size], grid_sizes[grid_size], num_steps, hardcode_configs[hc_config]); // re-route to pipelined implementation in pipeline.c
        MPI_Finalize();
        return EXIT_SUCCESS;
    default:
        return EXIT_FAILURE;
    }
#ifdef DEBUG
    size_t grid_height = NUM_ROWS_DEBUG;
    size_t grid_width = NUM_COLS_DEBUG;
#else
    size_t grid_height = (weak_scaling) ? grid_sizes[grid_size] / MAX_RANKS * world_size : grid_sizes[grid_size];
    size_t grid_width = grid_sizes[grid_size];
#endif
    if (!get_view_fn(&view, &neighbors, grid_height, grid_width))
    {
        fprintf(stderr,
                "Invalid number of ranks for %s fragmentation strategy\n",
                strategies[strategy]);
        return EXIT_FAILURE;
    }
    cuda_init_gridview(&view, world_rank);
    memset(view.grid.data, 0, view.grid.width * view.grid.height * sizeof(Cell_t));
    memset(view.next_grid.data, 0, view.next_grid.width * view.next_grid.height * sizeof(Cell_t));
#ifdef DEBUG
    // Test border exchange
    char fname[64];
    snprintf(fname, 64, "view%02d.txt", world_rank);
    FILE *f = fopen(fname, "w");
    fprintf(f, "Rank %02d:\n", world_rank);
    fprintf(f, "- row start: %zu\n", view.row_start);
    fprintf(f, "- col start: %zu\n", view.col_start);
    fprintf(f, "- height: %zu\n", view.height);
    fprintf(f, "- width: %zu\n", view.width);
    fprintf(f, "Exchange test:\n");
    for (int i = 1; i <= view.height; i++)
        for (int j = 1; j <= view.width; j++)
            row_ptr(&view.grid, i)[j] = world_rank + 1;
    fprintf(f, "Initial buffer:\n");
    for (int i = 0; i < view.grid.height; i++)
    {
        for (int j = 0; j < view.grid.width; j++)
            fprintf(f, "%02hhd ", row_ptr(&view.grid, i)[j]);
        fprintf(f, "\n");
    }
    exchange_fn(&view, &neighbors);
    fprintf(f, "Exchanged buffer:\n");
    for (int i = 0; i < view.grid.height; i++)
    {
        for (int j = 0; j < view.grid.width; j++)
            fprintf(f, "%02hhd ", row_ptr(&view.grid, i)[j]);
        fprintf(f, "\n");
    }
    memset(view.grid.data, 0, view.grid.width * view.grid.height * sizeof(Cell_t));
#endif
    uint64_t load_start, load_end;
    switch (hc_config)
    {
    case HC_CONFIG_ACORN:
        init_view(&view, &ACORN[0][0], 0, 0, 3, 7);
        break;
    case HC_CONFIG_BEACON:
        init_view(&view, &BEACON[0][0], 0, 0, 4, 4);
        break;
    case HC_CONFIG_BEEHIVE:
        init_view(&view, &BEEHIVE[0][0], 0, 0, 3, 4);
        break;
    case HC_CONFIG_GLIDER:
        init_view(&view, &GLIDER[0][0], 0, 0, 3, 3);
        break;
    case HC_CONFIG_TRAFFIC_LIGHT:
        init_view(&view, &TRAFFIC_LIGHT[0][0], 0, 0, 2, 3);
        break;
    case HC_CONFIG_NONE:
        MPI_Barrier(MPI_COMM_WORLD);
        load_start = clock_now();
        load_grid_view(&view, grid_height, grid_width, MPI_COMM_WORLD, load_checkpoint);
        load_end = clock_now();
        break;
    default:
        return EXIT_FAILURE;
    }
    exchange_fn(&view, &neighbors);
#ifdef DEBUG
    // Test initializer
    fprintf(f, "Simulation test:\n");
    fprintf(f, "Initial buffer:\n");
    for (int i = 0; i < view.grid.height; i++)
    {
        for (int j = 0; j < view.grid.width; j++)
            fprintf(f, "%hhd ", row_ptr(&view.grid, i)[j]);
        fprintf(f, "\n");
    }
#endif

    // Run simulation
    uint64_t start, end;
    MPI_Barrier(MPI_COMM_WORLD);
    start = clock_now();
    for (unsigned long i = 0; i < num_steps; i++)
    {
        // Run kernel
        run_kernel_nowrap(&view);
        // Swap grid and next_grid
        Cell_t *tmp = view.grid.data;
        view.grid.data = view.next_grid.data;
        view.next_grid.data = tmp;
        // Exchange border cells
        exchange_fn(&view, &neighbors);
#ifdef DEBUG
        fprintf(f, "Buffer %lu:\n", i + 1);
        for (int i = 0; i < view.grid.height; i++)
        {
            for (int j = 0; j < view.grid.width; j++)
                fprintf(f, "%hhd ", row_ptr(&view.grid, i)[j]);
            fprintf(f, "\n");
        }
#endif
    }
    end = clock_now();

#ifdef DEBUG
    fclose(f);
#endif

    // TODO save results to file
    uint64_t save_start, save_end;
    if (save_checkpoint != NULL)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        save_start = clock_now();
        save_grid_view(&view, grid_height, grid_width, MPI_COMM_WORLD, save_checkpoint);
        save_end = clock_now();
    }

    free_cudamem_gridview(&view);

    if (!world_rank)
    {
        printf("Results:\n");
        printf("- %lu simulations\n", num_steps);
        printf("- grid size %zu%s\n", grid_sizes[grid_size], (weak_scaling) ? " (weak scaling)" : "");
        printf("- %d ranks\n", world_size);
        printf("- %s strategy\n", strategies[strategy]);
        printf("- time taken to run: %.6f\n", (double)(end - start) / clock_frequency);
        if (load_checkpoint)
            printf("- load time %.6f\n", (double)(load_end - load_start) / clock_frequency);
        if (save_checkpoint)
            printf("- save time %.6f\n", (double)(save_end - save_start) / clock_frequency);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
