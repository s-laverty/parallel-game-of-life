#include <stdbool.h>
#include <stddef.h>

#define NUM_ROWS 32768 // 2^15
#define NUM_COLS 32768 // 2^15
#define WRAP_GRID true

/** Game of life cell data. Indexed as [row][col]. */
typedef bool **Grid;

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
inline size_t width(GridView *view)
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
inline size_t height(GridView *view)
{
#if WRAP_GRID
    return (NUM_ROWS + view->row_end - view->row_start) % NUM_ROWS;
#else
    return view->row_end - view->row_start;
#endif
}
