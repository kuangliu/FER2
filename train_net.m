function [net, loss_history] = train_net(net, X, y, X_val, y_val, opts)
%TRAIN_NET Train the network based on the given training data X.
% Inputs:
%   - X: training data [D, N]
%   - net: network struct
%   - opts: training options with:
%       - opts.lr: learning rat
%       - opts.reg: regularization strength
%       - opts.num_epochs
%       - opts.batch_size             
%
% Outputs:
%   - net: network after training with weight updated
%   - loss_history: recording the loss after each iteration [num_iters, 1]

layer_num = numel(net);
N = size(X, 2);


% Cache for Adam Update. 
% We first go through all the layers, if it's a weigted layer, then we 
% initialize ms&vs to zero
ms = cell(1, layer_num);
vs = cell(1, layer_num);
for layer_ind = 1:layer_num
    layer = net{layer_ind};
    if strcmp(layer.type, 'fc') || strcmp(layer.type, 'conv')   % weighted layer, init ms&vs
        ms{layer_ind} = zeros(size(layer.W), 'single');    
        vs{layer_ind} = zeros(size(layer.W), 'single');
    end
end

num_per_epoch = N / opts.batch_size;
num_iters = num_per_epoch * opts.num_epochs;
loss_history = zeros(num_iters, 1);

W_sum = 0;  % weight square sum for regularizaiotn
h = animatedline;   % draw lines for validation accuracy

% Training main loop
for it = 1:num_iters
    %  ----- sample a batch -----
    [X_batch, batch_idx] = datasample(X, opts.batch_size, 2, 'Replace', false);
    y_batch = y(batch_idx);
    % X_batch = X;
    % y_batch = y;
    
    %  ----- forward pass -----
    for layer_ind = 1:layer_num
        layer = net{layer_ind};

        switch layer.type
            case 'fc'
                net{layer_ind}.X = X_batch;      % save input for BP usage
                X_batch = fc_layer(layer.W, X_batch);   % forward X_batch through FC layer
                W_sum = W_sum + sum(sum(layer.W .* layer.W));   % add up all the sum of squared weights for regularization
            
            case 'relu'
                net{layer_ind}.X = X_batch;     % save input for BP usage
                X_batch = relu_layer(X_batch);
        end
    end
    
    %  ----- compute loss -----
    % grad is the output gradient, and we pass it back through the network
    % in the end, X_batch = final scores
    [loss, grad] = softmax_loss(X_batch, y_batch);
    
    % add regularization term
    loss_history(it) = loss + 0.5*opts.reg*W_sum;
    
    fprintf('#%d/%d: loss=%.4f\n', it, num_iters, loss)
    
    %  ----- backward pass -----
    %  compute local gradients
    for layer_ind = layer_num:-1:1
        layer = net{layer_ind};

        switch layer.type
            case 'fc'
                % output grad is the input gradient dX. we rename it to 'grad' for backprop convenience.
                [dW, grad] = fc_layer(layer.W, layer.X, grad);  
                
                % add regularization term
                dW = dW + opts.reg*layer.W;
                
                % perform Adam Update
                beta1 = 0.9;
                beta2 = 0.995;
                ms{layer_ind} = beta1 * ms{layer_ind} + (1-beta1) * dW;
                vs{layer_ind} = beta2 * vs{layer_ind} + (1-beta2) * (dW.*dW);
                mb = ms{layer_ind} ./ (1 - beta1^it);
                vb = vs{layer_ind} ./ (1 - beta2^it);
                net{layer_ind}.W = layer.W - opts.lr*mb./(sqrt(vb) + 1e-7);
            
            case 'relu'
                grad = relu_layer(layer.X, grad);
        end
    end
    
    % validation
    if mod(it, num_per_epoch) == 0
        val_acy = predict(net, X_val, y_val);
        addpoints(h, it/num_per_epoch, val_acy);
        drawnow limitrate
    end
end



