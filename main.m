clc;

N = 10;
D = 5;
C = 3;

X = randi(100, D, N, 'single');
X = bsxfun(@minus, X, mean(X,2));
X = bsxfun(@rdivide, X, std(X,0,2));

y = randi(C, 1, N, 'single');
W = randn(C, D, 'like', X) / sqrt(D);

a = fc_layer(W, X);

gamma = rand(C,1);
beta = zeros(C,1);
[b, a_norm] = bn_layer(a, gamma, beta);

[loss, grad] = svm_loss(b, y);

[grad, dGamma, dBeta] = bn_layer(a, a_norm, gamma, grad);

dW = fc_layer(W, X, grad);                                                                                          `

gradient_check(W, dW, X, y, gamma, beta)

