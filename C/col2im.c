#include "mex.h"

#define toIdx(a,b,c,d) ((a)+(b)*H+(c)*H*W+(d)*H*W*C)

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    /* input variable:
     *  - matrix M: [kH*kW*C, oH*oW*N]
     *  - input size: [H,W,C,N]
     *  - kernel size: [kH,kW]
     *  - output size: [oH,oW]
     *  - stride: S
     *
     * output variable: matrix of out_size=[H,W,C,N]
     *
     * e.g. im = col2im(M, [H,W,C,N], [kH,kW], [oH,oW], S);
     *
     */
    
    float* M = (float*)mxGetPr(prhs[0]); // [kH*kW*C, oH*oW*N]
    
    double *in_sz = mxGetPr(prhs[1]);
    int H = in_sz[0];
    int W = in_sz[1];
    int C = in_sz[2];
    int N = in_sz[3];
    const mwSize in_size[] = {H, W, C, N};
    
    double *kernel_sz = mxGetPr(prhs[2]);
    int kH = kernel_sz[0];
    int kW = kernel_sz[1];
    
    double *out_sz = mxGetPr(prhs[3]);
    int oH = out_sz[0];
    int oW = out_sz[1];

    int S = mxGetScalar(prhs[4]);
    
    plhs[0] = mxCreateNumericArray(4, in_size, mxSINGLE_CLASS, mxREAL);
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
}



