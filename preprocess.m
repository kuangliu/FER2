function X = preprocess(X)
%PREPROCESS Perform zero mean & normalization on X

% Zero mean
mean_img = mean(X, 2);
X = bsxfun(@minus, X, mean_img);

% Normalize
X = bsxfun(@rdivide, X, std(X));


