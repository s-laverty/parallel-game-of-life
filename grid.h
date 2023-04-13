/**
 * @file grid.h
 * @author Steven laverty (lavers@rpi.edu)
 * @brief This file defines game of life data structures and utility functions.
 * @version 0.1
 * @date 2023-04-09
 */

#ifndef _GRID_H
#define _GRID_H

#include <stdbool.h>
#include <string.h>

#ifndef DEBUG

#define NUM_ROWS 32768 // 2^15
#define NUM_COLS 32768 // 2^15
#define WRAP_GLOBAL_GRID true

typedef bool Cell_t;

#else

#define NUM_ROWS 13
#define NUM_COLS 13
#define WRAP_GLOBAL_GRID false

typedef char Cell_t;

#endif

/** A grid of binary cell data. */
typedef struct
{
    /** Game of life cell data. Flattened array of rows. */
    Cell_t *data;
    /** Number of rows. */
    size_t height;
    /** Number of columns. */
    size_t width;
} Grid;

/**
 * @brief Get a pointer to a grid row.
 *
 * @param grid The grid.
 * @param row_idx The index of the row.
 * @return Cell_t *
 */
inline Cell_t *row_ptr(Grid const *grid, size_t row_idx)
{
    return grid->data + (row_idx * grid->width);
}

/**
 * @brief Copy a grid row to a destination buffer.
 *
 * @param dest The destination buffer.
 * @param src The source grid.
 * @param row_idx The index of the row to copy from.
 * @param col_start The column index to start copying from.
 * @param length How many cells to copy.
 */
inline void get_row(Cell_t *dest, Grid const *src, size_t row_idx, size_t col_start, size_t length)
{
    memcpy(dest, row_ptr(src, row_idx) + col_start, length * sizeof(Cell_t));
}

/**
 * @brief Copy a source buffer to a grid row.
 *
 * @param dest The destination grid.
 * @param src The source buffer.
 * @param row_idx The index of the row to copy to.
 * @param col_start The column index to start copying to.
 * @param length How many cells to copy.
 */
inline void set_row(Grid *dest, Cell_t const *src, size_t row_idx, size_t col_start, size_t length)
{
    memcpy(row_ptr(dest, row_idx) + col_start, src, length * sizeof(Cell_t));
}

/**
 * @brief Copy a grid column to a destination buffer.
 *
 * @param dest The destination buffer.
 * @param src The source grid.
 * @param col_idx The index of the column to copy from.
 * @param row_start The row index to start copying from.
 * @param length How many cells to copy.
 */
inline void get_col(Cell_t *dest, Grid const *src, size_t col_idx, size_t row_start, size_t length)
{
    for (size_t i = 0; i < length; i++)
        dest[i] = row_ptr(src, row_start + i)[col_idx];
}

/**
 * @brief Copy a source buffer to a grid column.
 *
 * @param dest The destination grid.
 * @param src The source buffer.
 * @param col_idx The index of the column to copy to.
 * @param row_start The row index to start copying to.
 * @param length How many cells to copy.
 */
inline void set_col(Grid *dest, const Cell_t *src, size_t col_idx, size_t row_start, size_t length)
{
    for (size_t i = 0; i < length; i++)
        row_ptr(dest, row_start + i)[col_idx] = src[i];
}

/** A rectangular view of the global data grid with 1-cell padding. */
typedef struct
{
    /** Data in the grid view. */
    Grid grid;
    /** Output data. */
    Grid next_grid;
    /** Starting global row index of the grid view. */
    size_t row_start;
    /** Starting global column index of the grid view. */
    size_t col_start;
    /** Number of rows in the grid view (NOT including 1-cell padding). */
    size_t height;
    /** Number of columns in the grid view (NOT including 1-cell padding). */
    size_t width;
} GridView;

#endif
