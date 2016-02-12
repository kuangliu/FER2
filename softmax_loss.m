function [loss, dscores] = softmax_loss(scores, y)
%SOFTMAX_LOSS function
%
% Inputs have dimension D, there are C classes, and operate on minibatches
% of N examples.
%
% Inputs:
% - scores: [C, N]
% - y: [1, N]   make sure y is 1 based
%
% Returns a tuple of:
% - loss: single number of the svm loss
% - dscores: gradient with respect to scores


N = size(scores, 2);   % N samples

target_ind = sub2ind(size(scores), y, 1:N);

% compute loss, no regularization yet
probs = softmax(scores);
loss = -mean(log(probs(target_ind)));

% compute gradients
dscores = probs;
dscores(target_ind) = dscores(target_ind) - 1;
dscores = dscores / N;

% dW = dscores*X' + reg*W;








