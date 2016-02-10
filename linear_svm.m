% Linear Model: y = W*X using SGD to optimize.
load 'data.mat'

% Convert 'uint8' to 'single'
X_train = single(X_train);      % X_train: 3072*8000
X_val = single(X_val);          % X_val: 3072*2000

% Make labels from 0 based to 1 based
y_train = y_train + 1;  
y_val = y_val + 1;

classes = cellstr(['plane';'car  ';'bird ';'cat  ';'deer ';'dog  ';'frog ';'horse';'ship ';'truck']);
num_classes = numel(classes);

% Zero mean
mean_img = mean(X_train, 2);
imshow(reshape(uint8(mean_img), 32, 32, 3))

X_train = bsxfun(@minus, X_train, mean_img);
X_val = bsxfun(@minus, X_val, mean_img);

% Normalize
% If you do normalize after add bias term, remember reset them to 0 cause
% the bias has become NaN after @rdivide
X = bsxfun(@rdivide, X_train, std(X_train));


% Bias trick. Stack bias into the data.
%   Pro: We don't need to worry about bias anymore. 
%        Bias & weights are the same.
%   Cons: We cannot update weights&bias separately (maybe it's good...)
X = [X; zeros(1, size(X, 2))];                    % X: 3073*8000
X_val = [X_val; zeros(1, size(X_val, 2))];        % X_val: 3073*2000

C = 10;                     % C classes
[D, N] = size(X);     % N samples, each of D dimension
% Prepare the training parameters
opts.lr = 0.001;                    % learning rat
opts.reg = 0; %0.00001;              % regularization strength
opts.num_iters = 2000;               % # of iterations
opts.batch_size = 256;               

% Xavier initialization W
W = randn(C, D, 'like', X_train) / sqrt(D);

% Train the model
[W, loss_history] = train(W, X, y_train, opts);









