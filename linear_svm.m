%% Linear Model: y = W*X using SGD to optimize.
load 'data.mat'

% Convert 'uint8' to 'single'
X_train = single(X_train);      % X_train: 8000*3072
X_val = single(X_val);          % X_val: 2000*3072

% Make labels from 0 based to 1 based
y_train = y_train + 1;  
y_val = y_val + 1;

classes = cellstr(['plane';'car  ';'bird ';'cat  ';'deer ';'dog  ';'frog ';'horse';'ship ';'truck']);
num_classes = numel(classes);

% Zero mean
mean_img = mean(X_train);
imshow(reshape(uint8(mean_img), 32, 32, 3))

X_train = bsxfun(@minus, X_train, mean_img);
X_val = bsxfun(@minus, X_val, mean_img);

% Normalize
X = bsxfun(@rdivide, X_train, std(X));
X(:,end) = 0;   % after @rdivide 0, the bias has become NaN, reset them to 0

% Bias trick. Stack bias into the data.
%   Pro: We don't need to worry about bias anymore. 
%        Bias & weights are the same.
%   Cons: We cannot update weights&bias separately.
X_train = [X_train, zeros(size(X_train, 1), 1)];  % X_train: 8000*3073
X_val = [X_val, zeros(size(X_val, 1), 1)];        % X_val: 2000*3073

C = 10;                     % C classes
[N, D] = size(X_train);     % N samples, each of D dimension


% Prepare the training parameters
opts.lr = 0.001;                    % learning rat
opts.reg = 0; %0.00001;              % regularization strength
opts.num_iters = 2000;               % # of iterations
opts.batch_size = 256;               

% Xavier initialization W
W = randn(D, C, 'like', X_train)/sqrt(D);

% Train the model
[W, loss_history] = train(W, X, y_train, opts);









