function gradient_check(W, dW_analytic, X, y)
    h = 1e-6;
    num_checks = 100;       % We only check 'num_checks' gradients
    
    ind = randi(numel(W), num_checks, 1);
    for i = 1:num_checks
        W1 = double(W);
        W1(ind(i)) = W1(ind(i)) + h;
        loss1 = svm_loss(W1, X, y, 0);    
        
        W2 = double(W);
        W2(ind(i)) = W2(ind(i)) - h;
        loss2 = svm_loss(W2, X, y, 0);    
        
        dW_numerical = (loss1 - loss2) / (2 * h);
        dW_target = dW_analytic(ind(i));
        rel_error = abs(dW_numerical - dW_target) / (abs(dW_numerical) + abs(dW_target));
        fprintf('#%d, numerical: %f analytic: %f, relative error: %f\n', i, dW_numerical, dW_target, rel_error);
        
    end

end