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
%   - y: activations [oH,oW,kN,N]
%   - dX: input gradients [H,W,C,N]
%   - dW: weight gradients [kH,kW,C,kN]
%   - db: bias gradients [1, kN]

% use global variables for passing params easily.

if ~isfield(layer, 'stride')
    layer.stride = 1;
end

if ~isfield(layer, 'pad')
    layer.pad = 0; % if H=W, set pad to (kH-1)/2 to keep the input size,
end

if ~isa(layer.X, 'single')
    fprintf('Convert input data type to single.\n')
    layer.X = single(layer.X);
end

if ~isa(layer.W, 'single')
    fprintf('Convert weight data type to single.\n')
    layer.W = single(layer.W);
end

S = layer.stride;
P = layer.pad;

if nargin == 1 || isempty(varargin)
    % forward pass
    % Input size
    [H,W,C,N] = size(layer.X);
    [kH,kW,~,kN] = size(layer.W);

    % Output size
    oH = floor((H+2*P-kH)/S+1);
    oW = floor((W+2*P-kW)/S+1);

    if ~isfield(layer, 'input_size')
        layer.input_size = [H, W, C, N];
        layer.output_size = [oH, oW, kN, N];
    end
        
    % Padding
    layer.X = padarray(layer.X, [P,P]); % [H+2P,W+2P,C,N]
        
    weights = reshape(layer.W, kH*kW*C, kN);
    
    % The C version im2col is faster
    layer.M = im2col(layer.X, [kH,kW], [oH,oW], S);  % [kH*kW*C, oH*oW*N]
    %layer.M = im2col2(layer.X, [kH,kW], [oH,oW], S);  % [kH*kW*C, oH*oW*N]
    
    y = bsxfun(@plus, layer.M'*weights, layer.b);  % [oH*oW*N, kN]
    y = reshape(y, oH*oW, N, kN);
    y = permute(y, [1,3,2]);     % [oH*oW,kN,N]
    y = reshape(y, oH,oW,kN, N); % [oH,oW,kN,N]

    % output
    varargout{1} = y;
    varargout{2} = layer;
else
    % backward pass
    % Input size
    [HP,WP,C,N] = size(layer.X); % HP=H+2P, WP=W+2P
    [kH,kW,~,kN] = size(layer.W);

    % Output size
    oH = floor((HP-kH)/S+1);
    oW = floor((WP-kW)/S+1);
    
    dy = varargin{1};
    weights = reshape(layer.W, kH*kW*C, kN);
    
    dy = reshape(dy, oH*oW, kN, N);
    dy = permute(dy, [1,3,2]);     % [oH*oW, N, kN]
    dy = reshape(dy, oH*oW*N, kN); % [oH*oW*N, kN]
    
    dW = layer.M * dy;  % [kH*kW*C, kN]
    db = sum(dy);       % [1,kN]
    dM = weights * dy'; % [kH*kW*C, oH*oW*N]
    
    % The C version col2im is 10X faster than Matlab version
    % Note the size(X) is the size after padding
    dX = col2im(dM, size(layer.X), [kH,kW], [oH,oW], S); 
    %dX = col2im2(dM, [HP, WP, C, N], [kH, kW], [oH, oW], S); % Matlab version 
    
    % output
    varargout{1} = dX(P+1:end-P, P+1:end-P, :, :);
    varargout{2} = reshape(dW,kH,kW,C,kN);
    varargout{3} = db;
end


function M = im2col2(im, kernel_size, output_size, S)
% IM2COL convert a batch of images to cols for convolution
%
% In practice, use "im2col.c" instead. It's faster.
%
% Inputs:
%   - im: a batch of padded images sized [H+2P,W+2P,C,N]
%   - kernel_size: [kH, kW]
%   - output_size: [oH, oW]
%   - S: stride
%
% Output:
%   - M: a matrix sized [kH*kW*C, oH*oW*N], oH*oW is the # of receptive
%   fields of each image
%
[~,~,C,N] = size(im);
kH = kernel_size(1);
kW = kernel_size(2);
oH = output_size(1);
oW = output_size(2);

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

% with these crazy reshape trick, we can remove the (for n = 1:N) loop
M = reshape(M, kH*kW*C, N, oH*oW);
M = permute(M, [1,3,2]); % [kH*kW*C, oH*oW, N]
M = reshape(M, kH*kW*C, oH*oW*N);


function im = col2im2(M, im_size, kernel_size, output_size, S)
% COL2IM: convert column gradients back to original image gradients
%
% This function is re-implemented as "col2im.c". The logic is identical. 
% But C code is 10x faster.
%
% Inputs:
%   - M: sized [kH*kW*C, oH*oW*N]
%   - im_size: [H,W,C,N] original image size
%   - kernel_size: [kH,kW]
%   - output_size: [oH,oW]
%   - S: stride
%
% Outputs:
%   - im: the orignal padded image gradients, sized [H+2P,W+2P,C,N]
%
im = zeros(im_size, 'single');
C = im_size(3);
N = im_size(4);
kH = kernel_size(1);
kW = kernel_size(2);
oH = output_size(1);
oW = output_size(2);

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








