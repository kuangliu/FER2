function [net, loss_history] = train_net(X, y, net, opts)
%TRAIN_NET Train the network based on the given training data X.
% Inputs:
%   - X: training data [D, N]
%   - net: network struct
%   - opts: training options with:
%       - opts.lr: learning rat
%       - opts.reg: regularization strength
%       - opts.num_iters: # of iterations
%       - opts.batch_size             
%
% Outputs:
%   - net: network after training with weight updated
%   - loss_history: recording the loss after each iteration [num_iters, 1]

layer_num = numel(net);

loss_history = zeros(opts.num_iters, 1);

W_sum = 0;  % weight square sum for regularizaiotn

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

% Training main loop
for it = 1:opts.num_iters
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
        end
    end
    
    %  ----- compute loss -----
    % grad is the output gradient, and we pass it back through the network
    % in the end, X_batch = final scores
    [loss, grad] = svm_loss(X_batch, y_batch);
    
    % add regularization term
    loss_history(it) = loss + 0.5*opts.reg*W_sum;
    
    fprintf('#%d/%d: loss=%.4f\n', it, opts.num_iters, loss)
    
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
        end
    end
    
end



