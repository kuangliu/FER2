function [loss, dW] = softmax_loss(W, X, y, reg)
%SOFTMAX_LOSS function
%
% Inputs have dimension D, there are C classes, and operate on minibatches
% of N examples.
%
% Inputs:
% - W: [C, D]
% - X: [D, N]
% - y: [1, N]   make sure y is 1 based
% - reg: (float) regularization strength
%
% Returns a tuple of:
% - loss: single number of the svm loss
% - dW: gradient with respect to W, same shape as W


if ~isa(X, 'single')
    X = single(X);
end

[~, N] = size(X);   % N samples

% compute scores
scores = W * X;     % [C, N]
target_ind = sub2ind(size(scores), y, 1:N);

% compute loss
probs = softmax(scores);
loss = -mean(log(probs(target_ind))) + 0.5*reg*sum(sum(W.*W));

% compute gradients
dscores = probs;
dscores(target_ind) = dscores(target_ind) - 1;
dscores = dscores / N;

dW = dscores*X' + reg*W;








