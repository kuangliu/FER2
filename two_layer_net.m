function net = two_layer_net()
% A two layer FC neural network, with:
%   - input dimension: D
%   - hidden dimension: H
%   - output dimension: C
% We train the network with a svm loss function and L2 regularization.
% The network uses a ReLU nonlinearity after the first FC layer.
%
% The network structure:
%   input - fc - ReLU - fc - svmloss


D = 3073;
H = 200;
C = 10;

net = {};

% FC layer
%   - type: 'fc'
%   - W: weights
%   - X: input data for BP usage, added in training forward process
fc1.type = 'fc';
fc1.W = randn(H, D, 'single') / sqrt(D/2);
%fc1.W = [randn(H, D-1, 'single') / sqrt((D-1)/2), zeros(H, 1)];
net{end+1} = fc1;   % add to the network

% BN layer
bn1.type = 'bn';
bn1.gamma = rand(H,1); % uniform distribution
bn1.beta = zeros(H,1);
net{end+1} = bn1;

% ReLU layer
relu1.type = 'relu';
net{end+1} = relu1;

% FC layer
fc2.type = 'fc';
fc2.W = randn(C, H, 'single') / sqrt(H);
%fc2.W = [randn(C, H-1, 'single') / sqrt(H-1), zeros(C, 1)];
net{end+1} = fc2;





