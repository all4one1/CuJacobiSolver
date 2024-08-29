#include "CuSolver.h"

__global__ void swap_one(double* f_old, double* f_new, unsigned int N)
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < N)	f_old[l] = f_new[l];
}
__global__ void solveJacobiCuda(double* f, double* f0, double* b, int N, SparseMatrixCuda M)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	double s, diag;

	if (i < N)
	{
		s = 0;
		diag = 0;
		for (int j = M.row[i]; j < M.row[i + 1]; j++)
		{
			s += M.val[j] * f0[M.col[j]];
			if (M.col[j] == i) diag = M.val[j];
		}

		f[i] = f0[i] + (b[i] - s) / diag;
	}
}


__global__ void check(double* f, int N)
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < N; i++)
	{
		printf("%f ", f[i]);
	} printf("\n");
}



void CudaIterSolver::auto_test()
{
	//double A[6][6] =
	//{
	//	{ 30,3,4,0,0,0 },
	//	{ 4,22,1,3,0,0 },
	//	{ 5,7,33,6,7,0 },
	//	{ 0,1,2,42,3,3 },
	//	{ 0,0,2,11,52,2 },
	//	{ 0,0,0,3,9,26 },
	//};

	int nval = 24;
	int n = 6;
	double val[24] = { 30, 3, 4, 4, 22, 1, 3, 5, 7, 33, 6, 7, 1, 2, 42, 3, 3, 2, 11, 52, 2, 3, 9, 26 };
	int col[24] = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5 };
	int row[7] = { 0, 3, 7, 12, 17, 21, 24 };

	SparseMatrixCuda SMC(n, nval, val, col, row);


	double fh[6] = { 0, 0, 0, 0, 0, 0 };
	double* d, * d0, * b;
	cudaMalloc((void**)&d, sizeof(double) * n);
	cudaMalloc((void**)&d0, sizeof(double) * n);
	cudaMalloc((void**)&b, sizeof(double) * n);

	cudaMemcpy(d0, fh, sizeof(double) * n, cudaMemcpyHostToDevice);
	double bh[6] = { 1, 2, 3, 3, 2, 1 };
	cudaMemcpy(b, bh, sizeof(double) * n, cudaMemcpyHostToDevice);

	CudaLaunchSetup kernel(6);
	solveJacobi(d, d0, b, n, SMC, kernel);
	cudaMemcpy(fh, d, sizeof(double) * n, cudaMemcpyDeviceToHost);

	cout << "cuda test:   ";
	for (int i = 0; i < n; i++)
	{
		cout << fh[i] << " ";
	} cout << endl;

	double cg[6] =
	{ 0.1826929218e-1,
	0.7636750835e-1,
	0.5570467736e-1,
	0.6371099009e-1,
	0.2193724104e-1,
	0.2351661001e-1 };
	cout << "x should be: ";
	for (int i = 0; i < n; i++)
		cout << cg[i] << " ";
	cout << endl;


	cudaFree(d);
	cudaFree(d0);
	cudaFree(b);
}


CudaIterSolver::CudaIterSolver(){}

CudaIterSolver::CudaIterSolver(unsigned int N)
{
	CR = CudaReduction(N, 1024); 
}

void CudaIterSolver::solveJacobi(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel)
{
	CudaReduction CR(f, N, 1024);
	k = 0;
	eps = 1.0;
	res = 0.0;
	res0 = 0.0;

	for (k = 0; k < 200; k++)
	{
		solveJacobiCuda << < kernel.Grid1D, kernel.Block1D >> > (f, f0, b, N, M);

		//res = CR.reduce(f);
		res = CR.reduce();
		eps = abs(res - res0);
		res0 = res;

		swap_one << < kernel.Grid1D, kernel.Block1D >> > (f0, f, N);

		if (eps < eps_iter * res0) break;
	}

}