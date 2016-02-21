function [best_net, loss_history] = train_net(net, X, y, X_val, y_val, opts)
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

clc; close all;

layer_num = numel(net);
N = size(X, 2);

% Cache for Adam Update. 
% first go through all the layers, if it's a weigted layer, 
% then initialize ms&vs to zero
ms = cell(1, layer_num);
vs = cell(1, layer_num);
for layer_ind = 1:layer_num
    layer = net{layer_ind};
    if strcmp(layer.type, 'fc') || strcmp(layer.type, 'conv')    % weighted layer, init ms&vs
        ms{layer_ind} = zeros(size(layer.W), 'single');    
        vs{layer_ind} = zeros(size(layer.W), 'single');
    end
end

num_per_epoch = N / opts.batch_size;
num_iters = num_per_epoch * opts.num_epochs;
loss_history = zeros(num_iters, 1);

W_sum = 0;  % weight square sum for regularizaiotn
h = animatedline;  % for plotting validation accuracy

best_net = {};  % save the net with best val_acy
best_val_acy = 0;  % best validation accuracy

% Main loop of training
for it = 1:num_iters
    %  ------------------------------------------------------------------- 
    %                                                       sample a batch
    %  ------------------------------------------------------------------- 
    [X_batch, batch_idx] = datasample(X, opts.batch_size, 2, 'Replace', false);
    y_batch = y(batch_idx);
    % X_batch = X;
    % y_batch = y;
    
    % ------------------------------------------------------------------- 
    %                                                        forward pass
    % ------------------------------------------------------------------- 
    for layer_ind = 1:layer_num
        layer = net{layer_ind};

        switch layer.type
            case 'fc'
                % save input for BP usage
                net{layer_ind}.X = X_batch;
                % forward through FC layer
                X_batch = fc_layer(layer.W, X_batch);
                % add up all the sum of squared weights for regularization
                W_sum = W_sum + sum(sum(layer.W .* layer.W));
            
            case 'bn'
                net{layer_ind}.mode = 'train';
                net{layer_ind}.X = X_batch;
                % add running mean/std to bn layer
                [X_batch, net{layer_ind}] = bn_layer(net{layer_ind});
                
            case 'relu'
                % save input for BP usage
                net{layer_ind}.X = X_batch;
                % forward through ReLU layer
                X_batch = relu_layer(X_batch);
        end
    end
    
    % ------------------------------------------------------------------- 
    %                                             compute loss & gradient
    % ------------------------------------------------------------------- 
    % 'grad' is the output gradient, pass it back through the network
    % in the end: X_batch = final_scores
    [loss, grad] = svm_loss(X_batch, y_batch);
    
    % add regularization term
    loss_history(it) = loss + 0.5*opts.reg*W_sum;
    
    fprintf('#%d/%d: loss=%.4f\n', it, num_iters, loss)
    
    % ------------------------------------------------------------------- 
    %                                                       backward pass
    % ------------------------------------------------------------------- 
    for layer_ind = layer_num:-1:1
        layer = net{layer_ind};

        switch layer.type
            case 'fc'
                % output grad is the input gradient dX, renamed to 'grad' for BP convenience
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
            
            case 'bn'
                layer.mode = 'train';
                [grad, dGamma, dBeta] = bn_layer(layer, grad);
                % Vanilla Update gamma & beta
                net{layer_ind}.gamma = net{layer_ind}.gamma - 0.01*dGamma;
                net{layer_ind}.beta = net{layer_ind}.beta - 0.01*dBeta;
                
            case 'relu'
                grad = relu_layer(layer.X, grad);
                
        end
    end
    
    % validation
    if mod(it, num_per_epoch) == 0
        val_acy = predict(net, X_val, y_val);
        if val_acy > best_val_acy
            best_val_acy = val_acy;
            best_net = net;
        end
        % plot val_acy
        addpoints(h, it/num_per_epoch, val_acy);
        drawnow limitrate
    end
end



