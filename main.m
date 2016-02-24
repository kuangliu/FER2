function main()
clc;

C = 3;
D = 5;
N = 10;
X = randn(D, N);
y = randi(C, 1, 10);

layer.X = X;
layer.W = randn(C, D);
layer.b = zeros(C, 1);
a = fc_layer(layer);

[loss, dscores] = svm_loss(a, y);

[dX, dW, db] = fc_layer(layer, dscores);

gradient_check(db, layer, y)

% H = 5;
% W = 5;
% C = 3;
% N = 3;
% 
% X = randn(H,W,C,N,'single');
% y = [1,2,3];
% 
% layer.W = randn(3,3,3,3);
% 
% [a, layer1] = conv_layer(X, layer);
% a = reshape(a, [], 3);
% 
% [loss, dscores] = svm_loss(a, y);
% 
% da = reshape(dscores, 3, 3, 3, 3);
% [dX, dW] = conv_layer(X, layer1, da);
% 
% gradient_check(X, y, dW, layer1)



