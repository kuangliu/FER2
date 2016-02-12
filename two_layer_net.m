function net = two_layer_net(X, input_size, hidden_size, output_size)
%TWO_LAYER_NET A two-layer fully-connected neural network.
% Inputs:
%  - Input dimension:  N
%  - Hidden dimension: H
%  - Output dimension: C
%
% We train the network with softmax loss & L2 regularization.
% The network uses a ReLU nonlinearity after the first FC layer.
%
% Network structure:
%   input - FC - ReLU - FC - softmax

W1 = 0.1*randn(hidden_size, input_size, 'single');
b1 = zeros(hidden_size, 1);
W2 = 0.1*randn(output_size, hidden_size);
b2 = zeros(output_size, 1);

% forward
[loss, dscores] = forward(net, X);




end

function [loss, dscores] = forward(net, X)
% forward pass the network.

[D, N] = size(X);

a1 = bsxfun(@plus, W1*X, b1);
scores = bsxfun(@plus, W2*a1, b2);
    
    
end

