function [loss, accuracy] = predict(net, X, y_target)
%PREDICT the accuracy of trained network

layer_num = numel(net);
N = numel(y_target);

% forward the net to compute the scores
for i = 1:layer_num
    type = net{i}.type;
    
    switch type
        case 'fc'
            X = reshape(X, [], N);
            net{i}.X = X;
            X = fc_layer(net{i});
        case 'bn'
            net{i}.X = X;
            net{i}.mode = 'test';
            X = bn_layer(net{i});
        case 'relu'
            X = relu_layer(X);
        case 'conv'
            net{i}.X = X;
            [X, net{i}] = conv_layer_batch(net{i});
        case 'pool'
            net{i}.X = X;
            [X, net{i}] = pool_layer(net{i});
    end
end

loss = svm_loss(X, y_target);

[~, y_pred] = max(X);
num_correct = sum(y_pred==y_target);
accuracy = num_correct/size(X, 2);
fprintf('correct/total = %d/%d = %f\n', num_correct, size(X, 2), accuracy);
    
