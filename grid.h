#include <stdbool.h>
#include <stddef.h>

#define NUM_ROWS 32768 // 2^15
#define NUM_COLS 32768 // 2^15
#define WRAP_GRID false

/** A view of the global data grid. Indexed as [row][col]. */
typedef bool **Grid;

/** The bounds of a global data grid view. */
typedef struct
{
    /** Row index of the local grid view. */
    size_t row_start;
    /** Grid view continues up to, but not including, this row index. */
    size_t row_end;
    /** Column index of the local grid view. */
    size_t col_start;
    /** Grid view continues up to, but not including, this column index. */
    size_t col_end;
} GridDims;

/**
 * @brief Get the effective width of a grid view (including 1-cell padding).
 *
 * @param dims the grid view dimensions.
 * @return int
 */
inline int width(GridDims *dims)
{
    return dims->col_end - dims->col_start + 2;
}

/**
 * @brief Get the effective height of a grid view (including 1-cell padding).
 *
 * @param dims the grid view dimensions.
 * @return int
 */
inline int height(GridDims *dims)
{
    return dims->row_end - dims->row_start + 2;
}
