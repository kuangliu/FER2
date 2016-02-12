function net = build_net()
%BUILD_NET build the network layer by layer

C = 10;     % C classes
D = 3073;   % D dimension

net = {};

% 1. fc layer
% fc layer contains
%   - type: which is 'fc'
%   - W: weights
%   - X: input data, which cache for BP usage. And it added in training
%   forward process
fc1.type = 'fc';
fc1.W = randn(C, D, 'single') / sqrt(D);
net{end+1} = fc1;   % add to the network

    



