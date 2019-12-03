#include <stdio.h>
#include <stdlib.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
R A N K  A L G O R I T H M  R O U T I N E S
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define MATRIX_FORWARD_ELIMINATION 0
#define MATRIX_BACKWARD_ELIMINATION 1

void perform_elementary_row_operations( int flag,
                                        uint64_t i,
                                        uint64_t rows,
                                        uint64_t cols,
                                        std::vector<std::vector<bool>>& matrix )
{
    if( flag == MATRIX_FORWARD_ELIMINATION )
    {
        for( uint64_t j = i + 1; j < rows; j++ )
            if( matrix[j][i] == 1 )
                for( uint64_t k = i; k < cols; k++ )
                    matrix[j][k] = ( matrix[j][k] + matrix[i][k] ) % 2;
    }
    else
    {
        for( int64_t j = i - 1; j >= 0; j-- )
            if( matrix[j][i] == 1 )
                for( uint64_t k = 0; k < cols; k++ )
                    matrix[j][k] = ( matrix[j][k] + matrix[i][k] ) % 2;
    }
}

bool swap_rows( uint64_t i, uint64_t index, uint64_t cols, std::vector<std::vector<bool>>& matrix )
{
    for( uint64_t p = 0; p < cols; p++ )
    {
        bool temp = matrix[i][p];
        matrix[i][p] = matrix[index][p];
        matrix[index][p] = temp;
    }

    return true;
}

bool find_unit_element_and_swap( int flag, uint64_t i, uint64_t rows, uint64_t cols, std::vector<std::vector<bool>>& matrix )
{
    bool row_op = false;

    if( flag == MATRIX_FORWARD_ELIMINATION )
    {
        uint64_t index = i + 1;
        while( ( index < rows ) && ( matrix[index][i] == 0 ) )
            index++;
        if( index < rows )
            row_op = swap_rows( i, index, cols, matrix );
    }
    else
    {
        int64_t index = i - 1;
        while( ( index >= 0 ) && ( matrix[index][i] == 0 ) )
            index--;
        if( index >= 0 )
            row_op = swap_rows( i, index, cols, matrix );
    }

    return row_op;
}

int determine_rank( uint64_t m, uint64_t rows, uint64_t cols, const std::vector<std::vector<bool>>& matrix )
{
    int rank = m;
    for( uint64_t i = 0; i < rows; i++ )
    {
        bool allZeroes = true;
        for( uint64_t j = 0; j < cols; j++ )
        {
            if( matrix[i][j] )
            {
                allZeroes = false;
                break;
            }
        }
        if( allZeroes )
            rank--;
    }

    return rank;
}

int computeRank( uint64_t rows, uint64_t cols, std::vector<std::vector<bool>>& matrix )
{
    uint64_t m = rows < cols ? rows : cols;

    /* FORWARD APPLICATION OF ELEMENTARY ROW OPERATIONS */
    for( uint64_t i = 0; i < m - 1; i++ )
    {
        if( matrix[i][i] )
            perform_elementary_row_operations( MATRIX_FORWARD_ELIMINATION, i, rows, cols, matrix );
        else
        {
            if( find_unit_element_and_swap( MATRIX_FORWARD_ELIMINATION, i, rows, cols, matrix ) )
                perform_elementary_row_operations( MATRIX_FORWARD_ELIMINATION, i, rows, cols, matrix );
        }
    }

    /* BACKWARD APPLICATION OF ELEMENTARY ROW OPERATIONS */
    for( uint64_t i = m - 1; i > 0; i-- )
    {
        if( matrix[i][i] )
            perform_elementary_row_operations( MATRIX_BACKWARD_ELIMINATION, i, rows, cols, matrix );
        else
        {
            if( find_unit_element_and_swap( MATRIX_BACKWARD_ELIMINATION, i, rows, cols, matrix ) )
                perform_elementary_row_operations( MATRIX_BACKWARD_ELIMINATION, i, rows, cols, matrix );
        }
    }

    return determine_rank( m, rows, cols, matrix );
}

std::vector<std::vector<bool>> create_matrix( int rows, int cols )
{
    std::vector<std::vector<bool>> res;
    for( uint64_t row = 0; row < rows; row++ )
    {
        res.emplace_back( cols, false );
    }
    return res;
}


void populate_matrix( int rows, int cols, int matrix_id, const std::vector<bool>& input, std::vector<std::vector<bool>>& output )
{
    uint64_t start_idx = matrix_id * rows * cols;
    for( uint64_t row = 0; row < rows; row++ )
    {
        for( uint64_t col = 0; col < cols; col++ )
        {
            output[row][col] = input[start_idx + row * cols + col];
        }
    }
}

std::vector<std::vector<bool>> create_and_populate_matrix( int rows, int cols, int matrix_id, const std::vector<bool>& input )
{
    std::vector<std::vector<bool>> res;
    uint64_t start_idx = matrix_id * rows * cols;
    for( uint64_t row = 0; row < rows; row++ )
    {
        res.emplace_back( cols, false );
        for( uint64_t col = 0; col < cols; col++ )
        {
            res[row][col] = input[start_idx + row * cols + col];
        }
    }
    return res;
}