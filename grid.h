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
 * @brief Get the effective width of a grid view (including 1-cell padding).
 *
 * @param view The grid view.
 * @return size_t
 */
inline size_t padded_width(GridView *view)
{
    return view->col_end - view->col_start + 2;
}

/**
 * @brief Get the effective height of a grid view (including 1-cell padding).
 *
 * @param view The grid view.
 * @return size_t
 */
inline size_t padded_height(GridView *view)
{
    return view->row_end - view->row_start + 2;
}
