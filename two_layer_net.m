function net = two_layer_net()
% A two layer FC neural network, with:
% - input dimension: D
% - hidden dimension: H
% - output dimension: C
% We train the network with a svm loss function and L2 regularization.
% The network uses a ReLU nonlinearity after the first FC layer.
%
% The network structure:
%   input - fc - ReLU - fc - svmloss

D = 3073;
H = 200;
C = 10;

net = {};

% 1. fc layer contains:
%     - type: 'fc'
%     - W: weights
%     - X: input data, which cache for BP usage. 
%          And it added in training forward process
fc1.type = 'fc';
fc1.W = randn(H, D, 'single') / sqrt(D/2);
%fc1.W = [randn(H, D-1, 'single') / sqrt((D-1)/2), zeros(H, 1)];
net{end+1} = fc1;   % add to the network

% 2. ReLU layer
relu1.type = 'relu';
net{end+1} = relu1;

% 3. fc layer
fc2.type = 'fc';
fc2.W = randn(C, H, 'single') / sqrt(H);
%fc2.W = [randn(C, H-1, 'single') / sqrt(H-1), zeros(C, 1)];
net{end+1} = fc2;





