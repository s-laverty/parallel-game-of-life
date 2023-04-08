#include <stdbool.h>
#include <string.h>

#define NUM_ROWS 32768 // 2^15
#define NUM_COLS 32768 // 2^15
#define WRAP_GRID true

/** Game of life cell data. Indexed as [row][col]. */
typedef bool **Grid;

/**
 * @brief Copy a grid row to a destination buffer.
 *
 * @param dest The destination buffer.
 * @param src The source grid.
 * @param row_idx The index of the row to copy from.
 * @param col_start The column index to start copying from.
 * @param col_end Copy up to, but not including this column index.
 */
inline void get_row(bool *dest, const Grid src, size_t row_idx, size_t col_start, size_t col_end)
{
    memcpy(dest, src[row_idx] + col_start, (col_end - col_start) * sizeof(bool));
}

/**
 * @brief Copy a source buffer to a grid row.
 *
 * @param dest The destination grid.
 * @param src The source buffer.
 * @param row_idx The index of the row to copy to.
 * @param col_start The column index to start copying to.
 * @param col_end Copy up to, but not including this column index.
 */
inline void set_row(Grid dest, const bool *src, size_t row_idx, size_t col_start, size_t col_end)
{
    memcpy(dest[row_idx] + col_start, src, (col_end - col_start) * sizeof(bool));
}

/**
 * @brief Copy a grid column to a destination buffer.
 *
 * @param dest The destination buffer.
 * @param src The source grid.
 * @param col_idx The index of the column to copy from.
 * @param row_start The row index to start copying from.
 * @param row_end Copy up to, but not including this row index.
 */
inline void get_col(bool *dest, const Grid src, size_t col_idx, size_t row_start, size_t row_end)
{
    for (size_t i = row_start; i < row_end; i++)
        dest[i - row_start] = src[i][col_idx];
}

/**
 * @brief Copy a source buffer to a grid column.
 *
 * @param dest The destination grid.
 * @param src The source buffer.
 * @param col_idx The index of the column to copy to.
 * @param row_start The row index to start copying to.
 * @param row_end Copy up to, but not including this row index.
 */
inline void set_col(Grid dest, const bool *src, size_t col_idx, size_t row_start, size_t row_end)
{
    for (size_t i = row_start; i < row_end; i++)
        dest[i][col_idx] = src[i - row_start];
}

/** A view of the global data grid. */
typedef struct
{
    /** Data in the grid view. */
    Grid grid;
    /** Row index of the local grid view. */
    size_t row_start;
    /** Grid view continues up to, but not including, this row index. */
    size_t row_end;
    /** Column index of the local grid view. */
    size_t col_start;
    /** Grid view continues up to, but not including, this column index. */
    size_t col_end;
} GridView;

/**
 * @brief Get the width of a grid view (not including 1-cell padding).
 *
 * @param view The grid view.
 * @return size_t
 */
inline size_t view_width(const GridView *view)
{
#if WRAP_GRID
    return (NUM_COLS + view->col_end - view->col_start) % NUM_COLS;
#else
    return view->col_end - view->col_start;
#endif
}

/**
 * @brief Get the height of a grid view (not including 1-cell padding).
 *
 * @param view The grid view.
 * @return size_t
 */
inline size_t view_height(const GridView *view)
{
#if WRAP_GRID
    return (NUM_ROWS + view->row_end - view->row_start) % NUM_ROWS;
#else
    return view->row_end - view->row_start;
#endif
}
