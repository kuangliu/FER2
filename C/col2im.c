#include "mex.h"

#define im(a,b,c,d) im[(a)+(b)*H+(c)*H*W+(d)*H*W*C]

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    /* input variable:
     *  - matrix M: [kH*kW*C, oH*oW*N]
     *  - input size: [H,W,C,N]
     *  - kernel size: [kH,kW]
     *  - output size: [oH,oW]
     *  - stride: S
     *
     * output variable: im, matrix of input size [H,W,C,N]
     *
     * e.g. im = col2im(M, [H,W,C,N], [kH,kW], [oH,oW], S);
     */
    
    float *M = (float *)mxGetPr(prhs[0]); // [kH*kW*C, oH*oW*N]
    
    double *input_size = mxGetPr(prhs[1]);
    int H = input_size[0];
    int W = input_size[1];
    int C = input_size[2];
    int N = input_size[3];
    const mwSize im_size[] = {H, W, C, N};
    
    double *kernel_size = mxGetPr(prhs[2]);
    int kH = kernel_size[0];
    int kW = kernel_size[1];
    
    double *output_size = mxGetPr(prhs[3]);
    int oH = output_size[0];
    int oW = output_size[1];

    int S = mxGetScalar(prhs[4]);
    
    plhs[0] = mxCreateNumericArray(4, im_size, mxSINGLE_CLASS, mxREAL);
    float *im = (float *)mxGetPr(plhs[0]);
    
    int i = 0;
    for (int n = 0; n < N; ++n) {
        for (int w = 0; w < oW; ++w) {
            int x = w*S;
            for (int h = 0; h < oH; ++h) {
                int y = h*S;
                for (int c = 0; c < C; ++c) {
                    for (int xx = 0; xx < kW; ++xx) {
                        for (int yy = 0; yy < kH; ++yy) {
                            im(y+yy, x+xx, c, n) += M[i];
                            ++i;
                        }
                    }
                }
            }
        }
    }
}



