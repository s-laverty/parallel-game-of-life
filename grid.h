#include <stdbool.h>
#include <stddef.h>

#define NUM_ROWS 32768 // 2^15
#define NUM_COLS 32768 // 2^15
#define WRAP_GRID true

/** Game of life cell data. Indexed as [row][col]. */
typedef bool **Grid;

/**
 * @brief Copy a grid column to a destination buffer.
 *
 * @param dest The destination buffer.
 * @param src The source grid.
 * @param height The height of the grid.
 * @param col_idx The index of the column to copy from.
 */
inline void get_col(bool *dest, const Grid src, size_t height, size_t col_idx)
{
    for (size_t i = 0; i < height; i++)
        dest[i] = src[i][col_idx];
}

/**
 * @brief Copy a source buffer to a grid column.
 *
 * @param dest The destination grid.
 * @param src The source buffer.
 * @param height The height of the grid.
 * @param col_idx The index of the column to copy to.
 */
inline void set_col(Grid dest, const bool *src, size_t height, size_t col_idx)
{
    for (size_t i = 0; i < height; i++)
        dest[i][col_idx] = src[i];
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
