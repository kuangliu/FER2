#include "mex.h"
#include "matrix.h"
#include "stdio.h"

int H,W,C,N,kH,kW,oH,oW,S;

#define toIdx(a,b,c,d) (a+(b)*H+(c)*H*W+d*H*W*C)

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
 	float* M = (float*)mxGetPr(prhs[0]); // [kH*kW*C, oH*oW*N]
  	int H = mxGetScalar(prhs[1]);
  	int W = mxGetScalar(prhs[2]);
  	int C = mxGetScalar(prhs[3]);
  	int N = mxGetScalar(prhs[4]);
  	int kH = mxGetScalar(prhs[5]);
  	int kW = mxGetScalar(prhs[6]);
  	int oH = mxGetScalar(prhs[7]);
  	int oW = mxGetScalar(prhs[8]);
  	int S = mxGetScalar(prhs[9]);
	
	mwSize dims[4] = {H,W,C,N};	
	
	plhs[0] = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
	float *ptr = (float *)mxGetPr(plhs[0]);

	int i = 0;
	for (int n = 0; n < N; ++n) {
		for (int w = 0; w < oW; ++w) {
			int x = w*S;
			for (int h = 0; h < oH; ++h) {
				int y = h*S;
				for (int c = 0; c < C; ++c) {
					for (int xx = 0; xx < kW; ++xx) {
						for (int yy = 0; yy < kH; ++yy) {
							ptr[toIdx(y+yy, x+xx, c, n)] += M[i];	
							++i;
						}
					}
				}			
			}
		}
	}
	

	
//	plhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
}



