function [loss, accuracy] = predict(net, X, y_target)
%PREDICT the accuracy of trained network

layer_num = numel(net);

% forward the net to compute the scores
for layer_ind = 1:layer_num
    type = net{layer_ind}.type;
    
    switch type
        case 'fc'
            net{layer_ind}.X = X;
            X = fc_layer(net{layer_ind});
        case 'bn'
            net{layer_ind}.X = X;
            net{layer_ind}.mode = 'test';
            X = bn_layer(net{layer_ind});
        case 'relu'
            X = relu_layer(X);
    end
end

loss = svm_loss(X, y_target);

[~, y_pred] = max(X);
num_correct = sum(y_pred==y_target);
accuracy = num_correct/size(X, 2);
fprintf('correct/total = %d/%d = %f\n', num_correct, size(X, 2), accuracy);
    
