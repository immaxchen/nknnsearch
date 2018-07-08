/*=================================================================
 *
 * NKNNSEARCH searches for N test-point's K nearest-neighbors
 *     it performs linear search with size-K buffer
 *     using cityblock distance as distance metric
 *
 * The calling syntax is:
 *
 *     [I, D] = nknnsearch(X, Y, K)
 *
 * X is PxN matrix, Y is PxM matrix, K is scalar
 * I is KxM matrix, D is KxM matrix
 *
 * Max Chen 2018
 *
 *=================================================================*/

#include <float.h>
#include <math.h>

#include "mex.h"

#define X_IN  prhs[0]
#define Y_IN  prhs[1]
#define K_IN  prhs[2]

#define I_OUT plhs[0]
#define D_OUT plhs[1]

void nknnsearch( int N, int M, int P, int K,
                 double X[], double Y[], double I[], double D[] )
{
    double dist;
    int    imax = 0, i, j, k;
    
    for (i = 0; i < K*M; i++)
    {
        D[i] = DBL_MAX;
    }
    
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            dist = 0;
            for (k = 0; k < P; k++)
            {
                dist += fabs(X[k+j*P]-Y[k+i*P]);
            }
            if (dist < D[imax+i*K])
            {
                I[imax+i*K] = j+1;
                D[imax+i*K] = dist;
                
                for (k = 0; k < K; k++)
                {
                    if (D[k+i*K] > D[imax+i*K]) imax = k;
                }
            }
        }
    }
    
    return;
}

void mexFunction( int nlhs,       mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    if (nrhs < 3)
        mexErrMsgTxt("Not enough input arguments.");
    
    if (nrhs > 3)
        mexErrMsgTxt("Too many input arguments.");
    
    if (nlhs > 2)
        mexErrMsgTxt("Too many output arguments.");
    
    if (!mxIsDouble(X_IN))
        mexErrMsgTxt("Input X must be a double.");
    
    if (!mxIsDouble(Y_IN))
        mexErrMsgTxt("Input Y must be a double.");
    
    if (!mxIsDouble(K_IN))
        mexErrMsgTxt("Input K must be a double.");
    
    if (mxIsComplex(X_IN))
        mexErrMsgTxt("Input X must be noncomplex.");
    
    if (mxIsComplex(Y_IN))
        mexErrMsgTxt("Input Y must be noncomplex.");
    
    if (mxIsComplex(K_IN))
        mexErrMsgTxt("Input K must be noncomplex.");
    
    int nRowsX = mxGetM(X_IN);
    int nColsX = mxGetN(X_IN);
    
    int nRowsY = mxGetM(Y_IN);
    int nColsY = mxGetN(Y_IN);
    
    int nRowsK = mxGetM(K_IN);
    int nColsK = mxGetN(K_IN);
    
    if (nRowsX != nRowsY)
        mexErrMsgTxt("Dimensions of input X, Y are not consistent.");
    
    if (!(nRowsK==1 && nColsK==1))
        mexErrMsgTxt("Input K must be a scalar.");
    
    int N = nColsX;
    int M = nColsY;
    int P = nRowsX;
    
    double *DBL_K = mxGetPr(K_IN);
    
    int K = (int) *DBL_K + 0.5;
    
    if (K > N)
        mexErrMsgTxt("Input K cannot greater than size of X.");
    
    I_OUT = mxCreateDoubleMatrix(K, M, mxREAL);
    D_OUT = mxCreateDoubleMatrix(K, M, mxREAL);
    
    double *X = mxGetPr(X_IN);
    double *Y = mxGetPr(Y_IN);
    double *I = mxGetPr(I_OUT);
    double *D = mxGetPr(D_OUT);
    
    nknnsearch(N,M,P,K,X,Y,I,D);
    
    return;
}
