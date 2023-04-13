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

#ifndef DEBUG

#define MPI_CELL_Datatype MPI_C_BOOL

#else

#define MPI_CELL_Datatype MPI_CHAR

#endif

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
    view->grid.height = view->height + 2;
    view->grid.width = view->width + 2;

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
        view->width = ((col_idx + 1 < num_brick_cols) ? width.quot : offset) + width_pad;
    }
#endif
    view->grid.height = view->height + 2;
    view->grid.width = view->width + 2;

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

int main(int argc, char *argv[])
{
    /** Fragmentation strategies. */
    enum strategy
    {
        STRAT_STRIPED,
        STRAT_BRICK,
        STRAT_ERR
    };
    static char const *strategies[] = {[STRAT_STRIPED] = "striped",
                                       [STRAT_BRICK] = "brick"};
    /** Union type for grid view neighbors. */
    union neighbors
    {
        GridViewNeighborsStriped striped;
        GridViewNeighborsBrick brick;
    };
    static char const *arg_parse_err = "Usage: %s [-l checkpoint] strategy\n";

    MPI_Init(&argc, &argv);

    // Parse input args
    char const *load_checkpoint = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "l")) != -1)
    {
        switch (opt)
        {
        case 'l':
            load_checkpoint = optarg;
            break;
        default:
            fprintf(stderr, arg_parse_err, argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (argc - optind < 1)
    {
        fprintf(stderr, arg_parse_err, argv[0]);
        return EXIT_FAILURE;
    }
    enum strategy strategy = 0;
    while (strcmp(argv[optind], strategies[strategy]))
        if (++strategy == STRAT_ERR)
        {
            fprintf(stderr,
                    "Fragmentation strategy \"%s\" invalid. Must be one of:\n",
                    argv[optind]);
            for (strategy = 0; strategy < STRAT_ERR; strategy++)
                fprintf(stderr, "  %s\n", strategies[strategy]);
            return EXIT_FAILURE;
        }

    // Initialize
    GridView view;
    union neighbors neighbors;
    void (*exchange_fn)(GridView *, union neighbors const *);
    switch (strategy)
    {
    case STRAT_STRIPED:
        neighbors.striped.comm = MPI_COMM_WORLD;
        if (!get_view_striped(&view, &neighbors.striped, NUM_COLS, NUM_ROWS))
            return EXIT_FAILURE;
        exchange_fn = (void (*)(GridView *, union neighbors const *))exchange_border_cells_striped;
        break;
    case STRAT_BRICK:
        neighbors.brick.comm = MPI_COMM_WORLD;
        if (!get_view_brick(&view, &neighbors.brick, NUM_COLS, NUM_ROWS))
            return EXIT_FAILURE;
        exchange_fn = (void (*)(GridView *, union neighbors const *))exchange_border_cells_brick;
        break;
    default:
        return EXIT_FAILURE;
    }

#ifdef DEBUG
    // Test border exchange
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    Cell_t *buf1 = (Cell_t *)calloc(view.grid.width * view.grid.height, sizeof(Cell_t));
    view.grid.data = buf1;
    for (int i = 1; i <= view.height; i++)
        for (int j = 1; j <= view.width; j++)
            row_ptr(&view.grid, i)[j] = world_rank + 1;
    char fname[64];
    snprintf(fname, 64, "view%02d.txt", world_rank);
    FILE *f = fopen(fname, "w");
    fprintf(f, "Rank %02d:\n", world_rank);
    fprintf(f, "- row start: %zu\n", view.row_start);
    fprintf(f, "- col start: %zu\n", view.col_start);
    fprintf(f, "- height: %zu\n", view.height);
    fprintf(f, "- width: %zu\n", view.width);
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
    fclose(f);
    free(buf1);
#endif

    MPI_Finalize();

    return EXIT_SUCCESS;
}
