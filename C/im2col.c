#include "mex.h"

#define im(a,b,c,d) im[(a)+(b)*H+(c)*H*W+(d)*H*W*C]
#define M(a,b) M[(a)+(b)*C*kH*kW]

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    /* inputs:
     *  - im: a batch of images sized [H,W,C,N]
	 *  - kH, kW: kernel size
     *  - oH, oW: output size
     *  - S: stride
     *
     * output variable: matrix M of size [kH*kW*C, oH*oW*N]
     *
     * e.g. im = im2col(im, [kH,kW], [oH,oW], S);
     */
    
    float *im = (float *)mxGetPr(prhs[0]);
    
    const mwSize *input_size = mxGetDimensions(prhs[0]);
    int H = input_size[0];
    int W = input_size[1];
    int C = input_size[2];
    int N = input_size[3];
    
   	double *kernel_size = mxGetPr(prhs[1]);
    int kH = kernel_size[0];
    int kW = kernel_size[1];

   	double *output_size = mxGetPr(prhs[2]);
    int oH = output_size[0];
    int oW = output_size[1];
    
    int S = mxGetScalar(prhs[3]);
        
    const mwSize out_size[] = {kH*kW*C, oH*oW*N};
    plhs[0] = mxCreateNumericArray(2, out_size, mxSINGLE_CLASS, mxREAL);
    float *M = (float *)mxGetPr(plhs[0]);
    
	for (int n = 0; n < N; ++n) {
		for (int w = 0; w < oW; ++w) {
			int x = w*S;
			for (int h = 0; h < oH; ++h) {
				int y = h*S;
				int mx = h + w*oH + n*oH*oW;
				for (int c = 0; c < C; ++c) {
					for (int xx = 0; xx < kW; ++xx) {
						for (int yy = 0; yy < kH; ++yy) {
							int my = yy + xx*kH + c*kH*kW;
							M(my, mx) = im(y+yy, x+xx, c, n);
						}
					}
				}
			}
		}
	}
}










