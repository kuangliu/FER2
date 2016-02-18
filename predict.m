function accuracy = predict(net, X, y_target)
%PREDICT the accuracy of trained network

layer_num = numel(net);

% forward the net to compute the scores
for layer_ind = 1:layer_num
    layer = net{layer_ind};
    
    switch layer.type
        case 'fc'
            X = fc_layer(layer.W, X);
        case 'bn'
            X = bn_layer(X, layer.gamma, layer.beta);
        case 'relu'
            X = relu_layer(X);
    end
end

[~, y_pred] = max(X);

num_correct = sum(y_pred==y_target);
accuracy = num_correct/size(X, 2);
fprintf('correct/total = %d/%d = %f\n', num_correct, size(X, 2), accuracy);
    
