#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <vector>


#include "CudaReduction/CuReduction.h"

using std::cout;
using std::endl;
using std::ofstream;

struct CudaLaunchSetup
{
	dim3 Grid3D, Block3D, Grid1D, Block1D;
	unsigned int thread_x = 8, thread_y = 8, thread_z = 8, thread_1D = 1024;

	CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned nz = 1)
	{
		Grid3D = dim3(
			(unsigned int)ceil((nx + 1.0) / thread_x),
			(unsigned int)ceil((ny + 1.0) / thread_y),
			(unsigned int)ceil((nz + 1.0) / thread_z));
		Block3D = dim3(thread_x, thread_y, thread_z);

		Grid1D = dim3((unsigned int)ceil((N + 0.0) / thread_1D));
		Block1D = thread_1D;

	};
};

struct SparseMatrixCuda
{
	/*		Compressed Sparse Row	 */

	int Nfull = 0;	// the input (linear) size of a matrix
	int nval = 0;	// number of non-zero elements
	int nrow = 0;	// number of rows
	size_t bytesVal = 0;
	size_t bytesCol = 0;
	size_t bytesRow = 0;
	double* val = nullptr;
	int* col = nullptr;
	int* row = nullptr;

	SparseMatrixCuda() {};
	SparseMatrixCuda(int N, int nv, double* v, int* c, int* r) : Nfull(N), nval(nv)
	{
		nrow = N + 1;
		bytesVal = nval * sizeof(double);
		bytesCol = nval * sizeof(int);
		bytesRow = nrow * sizeof(int);

		cudaMalloc((void**)&val, sizeof(double) * nval);
		cudaMalloc((void**)&col, sizeof(int) * nval);
		cudaMalloc((void**)&row, sizeof(int) * nrow);

		cudaMemcpy(val, v, bytesVal, cudaMemcpyHostToDevice);
		cudaMemcpy(col, c, bytesCol, cudaMemcpyHostToDevice);
		cudaMemcpy(row, r, bytesRow, cudaMemcpyHostToDevice);
	}
	~SparseMatrixCuda() {};
};



struct CudaIterSolver
{
	int k = 0, write_i = 0, k_limit = 1000000;
	double eps_iter = 1e-6;
	double res = 0, res0 = 0, eps = 0;
	CudaReduction CR;

	CudaIterSolver();
	CudaIterSolver(unsigned int N);



	void solveJacobi(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel);

	void solveJacobi_experimental(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel,
		int k_minimal_threshold = 10, int k_frequency = 100);

	void auto_test();
};

