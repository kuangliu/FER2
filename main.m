function main()
clc;

H = 5;
W = 5;
C = 3;
N = 3;

X = randn(H,W,C,N,'single');
y = [1,2,3];

layer.W = randn(3,3,3,3);

[a, layer1] = conv_layer(X, layer);
a = reshape(a, [], 3);

[loss, dscores] = svm_loss(a, y);

da = reshape(dscores, 3, 3, 3, 3);
[dX, dW] = conv_layer(X, layer1, da);

gradient_check(X, y, dX, layer)



