#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;

#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>


#include "CuSolver.h"



int main()
{
	CudaIterSolver CUsolver;
	CUsolver.auto_test();


	////SparseMatrixCuda SMC(SM.Nfull, SM.nval, SM.nraw, SM.val.data(), SM.col.data(), SM.raw.data());

	int N = 100;
	size_t Nbytes = sizeof(double) * N;
	double* f_host = new double[N];
	memset(f_host, 0, Nbytes);

	double* f_dev, * f0_dev;


	//cudaMemcpy(fh, d, sizeof(double) * N, cudaMemcpyDeviceToHost);
	//	solver.auto_test();

	return 0;
}


