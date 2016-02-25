function varargout = conv_layer_batch(layer, varargin)
%CONV_LAYER_BATCH convolution layer without looping through N.
%
% It performs:
%   Forward pass: [y, layer] = conv_layer_batch(layer)
%   Backward pass: [dX, dW, db] = conv_layer_batch(layer, dy)
%
% Inputs:
%   - layer: convolution layer, with
%       - X: inputs [H,W,C,N]
%       - W: conv weights [kH,kW,C,kN] (kN filters, each sized [kH,kW,C])
%       - b: bias [1, kN]
%       - pad: 0 padding num
%       - stride
%       - M: im2col results [kH*kW*C, oH*oW*N]
%   - dy: output gradients
%
% Outputs:
%   - y: activations [oH*oW*kN,N]
%   - dX: input gradients [H,W,C,N]
%   - dW: weight gradients [kH,kW,C,kN]
%   - db: bias gradients [1, kN]

% use global variables for passing params easily.
global H W C N kH kW kN oH oW S P

if ~isfield(layer, 'stride')
    layer.stride = 1;
end

if ~isfield(layer, 'pad')
    layer.pad = 0; % if H=W, set pad to (kH-1)/2 to keep the input size,
end

X = layer.X;

% Input size
[H,W,C,N] = size(X);
[kH,kW,~,kN] = size(layer.W);

S = layer.stride;
P = layer.pad;

% Output size
oH = floor((H+2*P-kH)/S+1);
oW = floor((W+2*P-kW)/S+1);

if ~isfield(layer, 'input_size')
    layer.input_size = [H, W, C, N];
    layer.output_size = [oH, oW, kN, N];
end

if nargin == 1 || isempty(varargin)
    % forward pass
    % Padding
    X = padarray(X, [P,P]); % [H+2P,W+2P,C]
    
    weights = reshape(layer.W, kH*kW*C, kN);
    layer.M = im2col(X);    % [kH*kW*C, oH*oW*N]
    
    y = bsxfun(@plus, layer.M'*weights, layer.b);  % [oH*oW*N, kN]
    y = reshape(y, oH*oW, N, kN);
    y = permute(y, [1,3,2]);     % [oH*oW, kN, N]
    y = reshape(y, oH*oW*kN, N); % [oH*oW*kN, N]

    % output
    varargout{1} = y;
    varargout{2} = layer;
else
    % backward pass
    dy = varargin{1};
    weights = reshape(layer.W, kH*kW*C, kN);
    
    dy = reshape(dy, oH*oW, kN, N);
    dy = permute(dy, [1,3,2]);     % [oH*oW, N, kN]
    dy = reshape(dy, oH*oW*N, kN); % [oH*oW*N, kN]
    
    dW = layer.M * dy;  % [kH*kW*C, kN]
    db = sum(dy);       % [1,kN]
    dM = weights * dy'; % [kH*kW*C, oH*oW*N]
    %dX = col2im2(dM);
    
    dX = col2im(dM, [H,W,C,N], [kH,kW], [oH,oW], S);
    
    % output
    varargout{1} = dX;
    varargout{2} = reshape(dW,kH,kW,C,kN);
    varargout{3} = db;
end


function M = im2col(im)
% IM2COL convert a batch of images to cols for convolution
%
% Inputs:
%   - im: a batch of images sized [H,W,C,N]
%   - kH, kW: kernel size
%   - oH, oW: output size
%   - S: stride
%
% Output:
%   - M: a matrix sized [kH*kW*C, oH*oW*N], oH*OW is the # of receptive
%   fields of each image
%

global kH kW oH oW C N S

M = zeros(kH*kW*C*N, oH*oW);
i = 1;
for w = 1:oW
    x = 1+(w-1)*S;
    for h = 1:oH
        y = 1+(h-1)*S;
        cube = im(y:y+kH-1, x:x+kW-1, :, :);
        
        M(:,i) = cube(:); % reshape to 1 column
        i = i+1;
    end
end

% some crazy reshape tricks
M = reshape(M, kH*kW*C, N, oH*oW);
M = permute(M, [1,3,2]); % [kH*kW*C, oH*oW, N]
M = reshape(M, kH*kW*C, oH*oW*N);


function im = col2im_never_use(M)
% COL2IM: convert column gradients back to original image gradients
%
% This function is re-implemented as "col2im.c". The logic is identical. 
% But C code is 10x faster.
%
% Inputs:
%   - M: sized [kH*kW*C, oH*oW*N]
%   - H,W,C: im size
%   - kH,kW: kernel size
%   - S: stride
%
% Outputs:
%   - im: the orignal image gradients, sized [H,W,C,N]
%

global H W C N kH kW oH oW S

im = zeros(H,W,C,N);
i = 1;
for n = 1:N
    for w = 1:oW
        x = 1+(w-1)*S;
        for h = 1:oH
            y = 1+(h-1)*S;
            
            col = M(:,i);                   % [kH*kW*C, 1]
            col = reshape(col, kH, kW, C);  % [kH,kW,C]
            
            % collect the gradients
            im(y:y+kH-1, x:x+kW-1, :, n) = im(y:y+kH-1, x:x+kW-1, :, n) + col;
            i = i+1;
        end
    end
end








