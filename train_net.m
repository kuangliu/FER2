function res = train_net(net, X, y, X_val, y_val, opts)
%TRAIN_NET Train the network based on the given training data X.
% Inputs:
%   - X: training data [D, N]
%   - net: network struct
%   - opts: training options with:
%       - lr
%       - reg
%       - num_epochs
%       - batch_size             
%
% Outputs: res with:
%   - best_net: network with best validation accuracy
%   - train_losses
%   - val_losses
%   - val_accuracies
%   - best_val_accuracy
%


layer_num = numel(net);
N = size(X, 2);

num_per_epoch = N / opts.batch_size;
num_iters = num_per_epoch * opts.num_epochs;

% results
res.train_losses = zeros(opts.num_epochs, 1);
res.val_losses = zeros(opts.num_epochs, 1);
res.val_accuracies = zeros(opts.num_epochs, 1);
res.best_val_accuracy = 0;
res.best_net = {};  % save the net with best val_acy

W_sum = 0;  % weight square sum for regularizaiotn

% init Adam Update states
for i = 1:layer_num
    switch net{i}.type
        case 'fc'
            net{i}.stateW = struct(); % state for weights
            net{i}.stateB = struct(); % state for bias
        case 'bn'  
            net{i}.stateG = struct(); % state for Gamma
            net{i}.stateB = struct(); % state for Beta
    end
end


%--------------------------------------------------------------------------
%                            Training Starts
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
    for i = 1:layer_num
        switch net{i}.type
            case 'fc'
                % save input for BP usage
                net{i}.X = X_batch;
                % forward through FC layer
                X_batch = fc_layer(net{i});
                % add up all the sum of squared weights for regularization
                W_sum = W_sum + sum(sum(net{i}.W .* net{i}.W));
            
            case 'bn'
                net{i}.mode = 'train';
                net{i}.X = X_batch;
                % add running mean/std to bn layer
                [X_batch, net{i}] = bn_layer(net{i});
                
            case 'relu'
                % save input for BP usage
                net{i}.X = X_batch;
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
    loss = loss + 0.5*opts.reg*W_sum;
    
    % save the training loss for each epoch
    epoch_ind = floor((it-1)/num_per_epoch) + 1;
    res.train_losses(epoch_ind) = loss; 
    
    fprintf('epoch %d/%d, iteration %d/%d, loss=%.4f, lr=%.4f\n', ...
                epoch_ind, opts.num_epochs, it, num_iters, loss, opts.lr)
    
    % --------------------------------------------------------------------- 
    %                                                       backward pass
    % --------------------------------------------------------------------- 
    for i = layer_num:-1:1
        switch net{i}.type
            case 'fc'
                % output grad is the input gradient dX, 
                % renamed to 'grad' for BP convenience
                [grad, dW, db] = fc_layer(net{i}, grad);  
                
                % add regularization term, note we don't regularize bias
                dW = dW + opts.reg*net{i}.W;
                
                % update weights
                [net{i}.W, net{i}.stateW] = ...
                    adam_update(net{i}.W, dW, opts.lr, net{i}.stateW);
                
                % update bias
                [net{i}.b, net{i}.stateB] = ...
                    adam_update(net{i}.b, db, opts.lr, net{i}.stateB);
                 
            case 'bn'
                net{i}.mode = 'train';
                [grad, dGamma, dBeta] = bn_layer(net{i}, grad);
                
                % update gamma
                [net{i}.gamma, net{i}.stateG] = ...
                    adam_update(net{i}.gamma, dGamma, opts.lr, net{i}.stateG);
                
                % update beta
                [net{i}.beta, net{i}.stateB] = ...
                    adam_update(net{i}.beta, dBeta, opts.lr, net{i}.stateB);
                
            case 'relu'
                grad = relu_layer(net{i}.X, grad);
                
        end
    end
    
    % End of an epoch
    if mod(it, num_per_epoch) == 0 
        % weight decay
        if mod(epoch_ind, 30) == 0
            % every 30 epochs, lr decrease to 1/10
            opts.lr = opts.lr * 0.1;
        end
        
        % validation
        [res.val_losses(epoch_ind), res.val_accuracies(epoch_ind)] = ...
            predict(net, X_val, y_val);
        
        if res.val_accuracies(epoch_ind) > res.best_val_accuracy
            res.best_val_accuracy = res.val_accuracies(epoch_ind);
            res.best_net = net;
        end
        
        % plot results
        plot_info(res, epoch_ind)
        drawnow limitrate
    end
end



