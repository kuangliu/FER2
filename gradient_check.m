function gradient_check(W, dW_analytic, X, y, gamma, beta)
    h = 1e-3;
    num_checks = 50;       % We only check 'num_checks' gradients
    
    ind = randi(numel(W), num_checks, 1);
    for i = 1:num_checks
        W1 = double(W);
        W1(ind(i)) = W1(ind(i)) + h;
        a1 = fc_layer(W1, X);
        [b1, ~] = bn_layer(a1, gamma, beta);
        [loss1, ~] = svm_loss(b1, y);
        
        W2 = double(W);
        W2(ind(i)) = W2(ind(i)) - h;
        a2 = fc_layer(W2, X);
        [b2, ~] = bn_layer(a2, gamma, beta);
        [loss2, ~] = svm_loss(b2, y);
        
        dW_numerical = (loss1 - loss2) / (2 * h);
        dW_target = dW_analytic(ind(i));
        rel_error = abs(dW_numerical - dW_target) / (abs(dW_numerical) + abs(dW_target));
        fprintf('#%d, numerical: %f analytic: %f, relative error: %f\n', i, dW_numerical, dW_target, rel_error);
        
    end

end