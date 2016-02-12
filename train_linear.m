function [W, loss_history] = train_linear(W, X, y, opts)
%TRAIN_LINEAR the model using SGD.
loss_history = zeros(opts.num_iters,1);
m = zeros(size(W), 'like', W);    % cache for Adam update
v = zeros(size(W), 'like', W);

for it = 1:opts.num_iters
    % sample a batch
    [X_batch, batch_idx] = datasample(X, opts.batch_size, 2, 'Replace', false);
    y_batch = y(batch_idx);

    % evaluate loss and gradient
    [loss, dscores] = svm_loss(W*X_batch, y_batch);
    
    % add regularization term
    loss_history(it) = loss + 0.5*opts.reg*sum(sum(W.*W));

    % compute the local gradients, add regularization term here
    dW = dscores*X_batch' + opts.reg*W;
    
    % perform parameter update, we use Adam update
    beta1 = 0.9;
    beta2 = 0.995;
    m = beta1 * m + (1-beta1) * dW;
    v = beta2 * v + (1-beta2) * (dW.*dW);
    mb = m ./ (1 - beta1^it);
    vb = v ./ (1 - beta2^it);
    W = W - opts.lr*mb./(sqrt(vb) + 1e-7);

    fprintf('#%d/%d: loss=%.4f\n', it, opts.num_iters, loss)
end


