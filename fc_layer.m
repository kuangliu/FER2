function varargout = fc_layer(layer, varargin)
%FC_LAYER fully connected layer
%
% It performs:
%   - forward pass: y = fc_layer(layer)
%   - backward pass: [dX, dW, db] = fc_layer(layer, dy)
% 
% Inputs:
%   - layer: FC layer with
%       - X: input data, the size could be
%           - [D, N] as normal neural networks
%           - [H,W,C,N] if the previous layer is conv_layer
%       - W: weights [C, D]
%       - b: bias [C, 1]
%   - varargin: when perform backward pass, it is output gradient dy
%
% Outputs:
%   - varargout
%       - forward pass: return activations y = W*X+b
%       - backward pass: return local gradients dX, dW and db

X = layer.X;

% if X sized [H,W,C,N], reshape to [H*W*C,N]
if ndims(X) == 4
    X = reshape(X, [], size(X,4));
end

if nargin == 1 || isempty(varargin)
    % forward pass, compute activations y
    varargout{1} = bsxfun(@plus, layer.W*X, layer.b);
else
    % backward pass, compute local gradients dW & dX
    dy = varargin{1};
    dX = layer.W' * dy;                        % dX sized [D, N] 
    varargout{1} = reshape(dX, size(layer.X)); % reshape the gradient as the shape of X
    varargout{2} = dy * X';                    % dW
    varargout{3} = sum(dy, 2);                 % db
end




