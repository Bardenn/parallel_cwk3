// Kernel for matrix transposition.
__kernel
void transpose(__global const float* in, __global float* out, const int nRows, const int nCols) {
    // Get the row and column.
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // Check if the indices are within bounds.
    out[col * nRows + row] = in[row * nCols + col];
}