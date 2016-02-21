% function gradient_check(a, gamma, beta, dBeta, y)
% h = 1e-2;
% num_checks = numel(beta);
% 
% for i = 1:num_checks
%     beta1 = double(beta);
%     beta1(i) = beta1(i) + h;
%     [b1, ~] = bn_layer(a, gamma, beta1);
%     [loss1, ~] = svm_loss(b1, y);
%     
%     beta2 = double(beta);
%     beta2(i) = beta2(i) - h;
%     [b2, ~] = bn_layer(a, gamma, beta2);
%     [loss2, ~] = svm_loss(b2, y);
%     
%     dG_numerical = (loss1 - loss2) / (2 * h);
%     dG_target = dBeta(i);
%     rel_error = abs(dG_numerical - dG_target) / ...
%                     (abs(dG_numerical) + abs(dG_target));
%     fprintf('#%d, numerical: %f analytic: %f, relative error: %f\n',...
%                 i, dG_numerical, dG_target, rel_error);
% end
% 



function gradient_check(W, dW_analytic, X, y, layer)
    h = 1e-2;
    num_checks = 25;       % We only check 'num_checks' gradients
    
    for i = 1:num_checks
        W1 = double(W);
        W1(i) = W1(i) + h;
        a1 = fc_layer(W1, X);
        layer1 = layer;
        layer1.X = a1;
        [b1, ~] = bn_layer(layer1);
        [loss1, ~] = svm_loss(b1, y);
        
        W2 = double(W);
        W2(i) = W2(i) - h;
        a2 = fc_layer(W2, X);
        layer2 = layer;
        layer2.X = a2;
        [b2, ~] = bn_layer(layer2);
        [loss2, ~] = svm_loss(b2, y);
        
        dW_numerical = (loss1 - loss2) / (2 * h);
        dW_target = dW_analytic(i);
        rel_error = abs(dW_numerical - dW_target) / ...
                      (abs(dW_numerical) + abs(dW_target));
        fprintf('#%d, numerical: %f analytic: %f, relative error: %f\n', ...
                      i, dW_numerical, dW_target, rel_error);
        
    end

end