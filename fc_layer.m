function varargout = fc_layer(W, X, varargin)
%FC_LAYER fully connected layer
%
% It performs:
%   - forward pass: y = fc_layer(W, X)
%   - backward pass: [dW, dX] = fc_layer(W, X, dy)
% 
% Inputs:
%   - W: weights [C, D]
%   - X: input data [D, N]
%   - varargin: output gradient dy, when we perform forward pass, it's null
%
% Outputs:
%   - varargout
%       - forward pass: activations y = W * X
%       - backward pass: local gradients dW & dX


if nargin == 2 || isempty(varargin)
    % forward pass, compute activations y
    varargout{1} = W * X;
else
    % backward pass, compute local gradients dW & dX
    dy = varargin{1};
    varargout{1} = dy * X';     % dW
    varargout{2} = W' * dy;     % dX
end




