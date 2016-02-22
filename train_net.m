function [best_net, info] = train_net(net, X, y, X_val, y_val, opts)
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
%   - best_net: network with best validation accuracy
%   - info containing
%       - train_losses
%       - val_losses
%       - val_accuracies


layer_num = numel(net);
N = size(X, 2);

num_per_epoch = N / opts.batch_size;
num_iters = num_per_epoch * opts.num_epochs;

% info
info.train_losses = zeros(num_iters, 1);
info.val_losses = zeros(opts.num_epochs, 1);
info.val_accuracies = zeros(opts.num_epochs, 1);

W_sum = 0;  % weight square sum for regularizaiotn
states = cell(1, layer_num); % Adam Update status
best_net = {};  % save the net with best val_acy

%--------------------------------------------------------------------------
%                            Training Main Loop
%--------------------------------------------------------------------------

for it = 1:num_iters
    %  --------------------------------------------------------------------
    %                                                       sample a batch
    %  --------------------------------------------------------------------
    [X_batch, batch_idx] = datasample(X, opts.batch_size, 2, 'Replace', false);
    y_batch = y(batch_idx);
    % X_batch = X;
    % y_batch = y;
    
    % --------------------------------------------------------------------- 
    %                                                        forward pass
    % --------------------------------------------------------------------- 
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
    
    % --------------------------------------------------------------------- 
    %                                             compute loss & gradient
    % --------------------------------------------------------------------- 
    % 'grad' is the output gradient, pass it back through the network
    % in the end: X_batch = final_scores
    [loss, grad] = svm_loss(X_batch, y_batch);
    
    % add regularization term
    info.train_losses(it) = loss + 0.5*opts.reg*W_sum;
    
    epoch_ind = floor((it-1)/num_per_epoch) + 1;
    fprintf('epoch %d/%d, iteration %d/%d, loss=%.4f, lr=%.4f\n', ...
                epoch_ind, opts.num_epochs, it, num_iters, loss, opts.lr)
    
    % --------------------------------------------------------------------- 
    %                                                       backward pass
    % --------------------------------------------------------------------- 
    for layer_ind = layer_num:-1:1
        layer = net{layer_ind};

        switch layer.type
            case 'fc'
                % output grad is the input gradient dX, renamed to 'grad' for BP convenience
                [dW, grad] = fc_layer(layer.W, layer.X, grad);  
                
                % add regularization term
                dW = dW + opts.reg*layer.W;
                
                % Adam Update
                [net{layer_ind}.W, states{layer_ind}] = ...
                    adam_update(layer.W, dW, opts.lr, states{layer_ind});
            
            case 'bn'
                layer.mode = 'train';
                [grad, dGamma, dBeta] = bn_layer(layer, grad);
                
                % Adam Update gamma & beta
                % 1. cat gamma & beta together sized [D,2]
                params = [layer.gamma, layer.beta];
                dParams = [dGamma, dBeta];
                
                % 2. Adam update cated params
                [params, states{layer_ind}] = ...
                    adam_update(params, dParams, opts.lr, states{layer_ind});
                
                % 3. split updated params back to gamma & beta
                net{layer_ind}.gamma = params(:,1);
                net{layer_ind}.beta = params(:,2);
                
            case 'relu'
                grad = relu_layer(layer.X, grad);
                
        end
    end
    
    % End of an epoch
    if mod(it, num_per_epoch) == 0 
        % 1. weight decay
        if mod(epoch_ind, 30) == 0
            % every 30 epochs, lr decrease to 1/10
            opts.lr = opts.lr * 0.1;
        end
        
        % 2. validation
        [info.val_losses(epoch_ind), info.val_accuracies(epoch_ind)] = ...
                            predict(net, X_val, y_val);
        plot_info(info, epoch_ind)  % plot results
        drawnow limitrate
    end
end



