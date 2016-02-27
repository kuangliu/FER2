function net = build_net()
% A two layer FC neural network, with:
%   - input dimension: D
%   - hidden dimension: H
%   - output dimension: C
% We train the network with a svm loss function and L2 regularization.
% The network uses a ReLU nonlinearity after the first FC layer.
%
% The network structure:
%   input - fc - ReLU - fc - svmloss

net = {};

% Conv layer
c1.type = 'conv';
c1.W = randn(3,3,3,32)/sqrt(3*3*3/2);
c1.b = zeros(1,32);
c1.stride = 1;
c1.pad = 1;
net{end+1} = c1;

% ReLU
r1.type = 'relu';
net{end+1} = r1;

% Conv layer
c2.type = 'conv';
c2.W = randn(3,3,32,64)/sqrt(3*3*32/2);
c2.b = zeros(1,64);
c2.stride = 1;
c2.pad = 1;
net{end+1} = c2;

% ReLU
r2.type = 'relu';
net{end+1} = r2;

% Conv layer
c3.type = 'conv';
c3.W = randn(3,3,64,64)/sqrt(3*3*64);
c3.b = zeros(1,64);
c3.stride = 2;
c3.pad = 1;
net{end+1} = c3;

% FC layer
f1.type = 'fc';
f1.out_size = 10;
net{end+1} = f1;

% FC layer
%   - type: 'fc'
%   - W: weights
%   - b: bias
%   - X: input data for BP usage, added in training forward process
%   - we can specify the W&b; beside we can only specify the out_size.
% fc1.type = 'fc';
% fc1.W = randn(H, D, 'single')/sqrt(D/2);
% fc1.b = zeros(H, 1);
% net{end+1} = fc1;   % add to the network

% BN layer
% bn1.type = 'bn';
% net{end+1} = bn1;

% ReLU layer
% relu1.type = 'relu';
% net{end+1} = relu1;

% FC layer
% fc2.type = 'fc';
% fc2.W = randn(C, H, 'single')/sqrt(H);
% fc2.b = zeros(C, 1);
% net{end+1} = fc2;





