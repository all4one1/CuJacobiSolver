#include "CuSolver.h"



int main()
{
	CudaIterSolver CUsolver;
	CUsolver.auto_test();

	// usage: 

	// rank of a square matrix 
	int n = 6; 
	// number of non-zero elements of a matrix
	int nval = 24;

	// Example of a sparse matrix
	double sparse_matrix_elements[24] = { 30, 3, 4, 4, 22, 1, 3, 5, 7, 33, 6, 7, 1, 2, 42, 3, 3, 2, 11, 52, 2, 3, 9, 26 };
	int column[24] = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5 };
	int row[7] = { 0, 3, 7, 12, 17, 21, 24 };
	

	SparseMatrixCuda SMC(n, nval, sparse_matrix_elements, column, row);
	CudaLaunchSetup kernel_settings(n);

	// double* f_dev, * f0_dev, * rhs_dev;
	// CUsolver.solveJacobi(f_dev, f0_dev, rhs_dev, n, SMC, kernel_settings);

	return 0;
}


