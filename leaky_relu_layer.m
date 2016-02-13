function y = leaky_relu_layer(X, varargin)
%LEAKY_RELU_LAYER
% - forward pass: y = leaky_relu_layer(X) = max(0.01*X, X)
% - backward pass: dy = leaky_relu_layer(X, dX) = dX.*(X>0) + 0.01.*dX.*(X<0)

if nargin == 1 || isempty(varargin) 
    % forward pass, y is activations
    y = max(0.01*X, X);
else 
    % backward pass, y is gradients
    dX = varargin{1};
    y = dX.*(X>0) + 0.01.*dX.*(X<=0);
end



