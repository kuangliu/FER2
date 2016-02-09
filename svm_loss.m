function [loss, dW] = svm_loss(W, X, y, reg)
%SVM loss function
%
% Inputs have dimension D, there are C classes, and operate on minibatches
% of N examples.
%
% Inputs:
% - W: [D, C]
% - X: [N, D]
% - y: [N,  ]   make sure y is 1 based
% - reg: (float) regularization strength
%
% Returns a tuple of:
% - loss: single number of the svm loss
% - dW: gradient with respect to W, same shape as W

if ~isa(X, 'single')
    X = single(X);
end

if ~isa(y, 'single')
    y = single(y);
end

[N, ~] = size(X);   % N samples

% compute scores
scores = X * W;     % [N, C]

% Index trick:
% To get the score of the target label, we first convert subscript to
% index, and get the element by index.
target_ind = sub2ind(size(scores), (1:N)', y);
target_scores = scores(target_ind);     % [N, 1]

% compute margins
margins = bsxfun(@minus, scores, target_scores) + 1;    % [N, C]

% hinge loss
hinge_loss = max(0, margins);    

% set the loss of target label to 0
hinge_loss(target_ind) = 0;

% loss = margin sum / N + regularization
loss = sum(sum(hinge_loss)) / N + 0.5*reg*sum(sum(W.*W));

% dscores = dloss/dscores
dscores = zeros(size(scores),'like',scores);
dscores(margins>0) = 1;
dscores(target_ind) = dscores(target_ind) - sum(margins>0, 2);
dscores = dscores / single(N);

% scores = W * X + 0.5*reg*W^2
% => dW = dscores * X + reg*W
dW = X'*dscores + reg*W;


