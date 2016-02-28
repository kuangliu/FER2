#include "mex.h"

#define X(a,b,c,d) X[(a)+(b)*H+(c)*H*W+(d)*H*W*C]
#define toIdx(a,b,c,d) ((a)+(b)*H+(c)*H*W+(d)*H*W*C)
#define M(a,b,c,d) M[(a)+(b)*oH+(c)*oH*oW+(d)*oH*oW*C]
#define I(a,b,c,d) I[(a)+(b)*oH+(c)*oH*oW+(d)*oH*oW*C]

const int INF = 987654321;

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    /* inputs:
     *  - X: input sized [H,W,C,N]
     *  - kH, kW: kernel size
     *  - S: stride
     *
     * output variable: 
     *  - value matrix: M of size [oH,oW,C,N]
     *  - index matrix: I of size [oH,oW,C,N]
     *
     * e.g. [y,inds] = maxpool(X, [kH,kW], S);
     */
    
    float *X = (float *)mxGetPr(prhs[0]);
    
    const mwSize *input_size = mxGetDimensions(prhs[0]);
    int H = input_size[0];
    int W = input_size[1];
    int C = input_size[2];
    int N = input_size[3];
    
    double *kernel_size = mxGetPr(prhs[1]);
    int kH = kernel_size[0];
    int kW = kernel_size[1];
    
    int S = mxGetScalar(prhs[2]);
    
    int oH = (H-kH)/S+1;
    int oW = (W-kW)/S+1;
    const mwSize M_size[] = {oH,oW,C,N};
    
    plhs[0] = mxCreateNumericArray(4, M_size, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(4, M_size, mxSINGLE_CLASS, mxREAL);
    
    float *M = (float *)mxGetPr(plhs[0]);
    float *I = (float *)mxGetPr(plhs[1]);
    
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int w = 0; w < oW; ++w) {
                int x = w*S;
                for (int h = 0; h < oH; ++h) {
                    int y = h*S;
                    int max_value = -INF;
                    int max_y = 0, max_x = 0;
                    for (int xx = 0; xx < kW; ++xx) {
                        for (int yy = 0; yy < kH; ++yy) {
                            if (X(y+yy, x+xx, c, n) > max_value) {
                                max_value = X(y+yy, x+xx, c, n);
                                max_y = y+yy;
                                max_x = x+xx;
                            }
                            
                        }
                    }
                    M(h,w,c,n) = max_value;
                    I(h,w,c,n) = toIdx(max_y,max_x,c,n) + 1; // 0 based to 1 based
                }
            }
        }
    }
    
    
}










