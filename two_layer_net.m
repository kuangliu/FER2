function net = two_layer_net()
%Build a 2 layer network

D = 3073;
H = 50;
C = 10;

net = {};

% fc layer
%   - type: which is 'fc'
%   - W: weights
%   - X: input data, which cache for BP usage. 
%        And it added in training forward process
fc1.type = 'fc';
fc1.W = randn(H, D, 'single') / sqrt(D);
net{end+1} = fc1;   % add to the network

fc2.type = 'fc';
fc2.W = randn(C, H, 'single') / sqrt(H);
net{end+1} = fc2;





