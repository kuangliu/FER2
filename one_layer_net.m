function net = one_layer_net()
%ONE_LAYER_NET one FC layer net
C = 10;     % C classes
D = 3073;   % D dimension, with bias

net = {};

% 1. fc layer
% fc layer contains
%   - type: which is 'fc'
%   - W: weights
%   - X: input data, which cache for BP usage. And it added in training
%   forward process
fc1.type = 'fc';
fc1.W = randn(C, D, 'single') / sqrt(D);       % init weights & bias the same
% fc1.W = [randn(C, D-1, 'single') / sqrt(D-1), zeros(C, 1)]; % init bias term to 0
net{end+1} = fc1;   % add to the network

    




