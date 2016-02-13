function y = relu_layer(X, varargin)
%RELU layer
% It computes:
% - forward pass: y = relu_layer(X) = max(0, X)
% - backward pass: dy = relu_layer(X, dX) = dX.*(X>0)

if nargin == 1 || isempty(varargin) 
    % forward pass, y is activations
    y = max(0, X);
else 
    % backward pass, y is gradients
    dX = varargin{1};
    y = dX.*(X>0);
end





