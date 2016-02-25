function main()
clc;

% C = 3;
% D = 5;
% N = 10;
% X = randn(D, N);
% y = randi(C, 1, 10);
%
% layer.X = X;
% layer.W = randn(C, D);
% layer.b = zeros(C, 1);
% a = fc_layer(layer);
%
% [loss, dscores] = svm_loss(a, y);
%
% [dX, dW, db] = fc_layer(layer, dscores);
%
% gradient_check(db, layer, y)

H = 5;
W = 5;
C = 3;
N = 3;

X = randn(H,W,C,N,'single');
y = [1,2,3];

layer.X = X;
layer.W = randn(3,3,3,3);
layer.b = zeros(1,3);

[a, layer] = conv_layer_batch(layer);
a = reshape(a, [], 3);

[loss, dscores] = svm_loss(a, y);

da = reshape(dscores, 3, 3, 3, 3);
[dX, dW, db] = conv_layer_batch(layer, da);
gradient_check(y, db, layer)



