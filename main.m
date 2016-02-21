clc;

N = 10;
D = 5;
C = 5;

X = randi(100, D, N, 'single');
X = bsxfun(@minus, X, mean(X,2));
X = bsxfun(@rdivide, X, std(X,0,2));

y = randi(C, 1, N, 'single');
W = randn(C, D, 'like', X) / sqrt(D);

a = fc_layer(W, X);

layer.gamma = rand(C,1);
layer.beta = zeros(C,1);
layer.X = a;
layer.mode = 'train';
layer.running_mean = zeros(C,1);
layer.running_std = ones(C,1);
[b, layer] = bn_layer(layer);

[loss, grad] = svm_loss(b, y);

[grad, dGamma, dBeta] = bn_layer(layer, grad);

[dW, dX] = fc_layer(W, X, grad);
gradient_check(W, dW, X, y, layer)



