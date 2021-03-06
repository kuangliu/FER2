function gradient_check(y, dX, layer)

h = 1e-3;    
num_checks = 81;
for i = 1:num_checks
    layer1 = layer;
    layer1.X(i) = layer1.X(i) + h;
    [a1, ~] = conv_layer(layer1);
    a1 = reshape(a1, [], 3);
    [loss1, ~] = svm_loss(a1, y);
    
    layer2 = layer;
    layer2.X(i) = layer2.X(i) - h;
    [a2, ~] = conv_layer(layer2);
    a2 = reshape(a2, [], 3);
    [loss2, ~] = svm_loss(a2, y);
    
    dNumerical = (loss1 - loss2) / (2 * h);
    dTarget = dX(i);
    rel_error = abs(dNumerical - dTarget) / ...
        (abs(dNumerical) + abs(dTarget));
    fprintf('#%d, numerical: %f analytic: %f, relative error: %f\n',...
        i, dNumerical, dTarget, rel_error);
end




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



% function gradient_check(dW, layer, y)
%     h = 1e-3;
%     num_checks = 3;       % We only check 'num_checks' gradients
%     
%     for i = 1:num_checks
%         layer1 = layer;
%         layer1.b(i) = layer1.b(i) + h;
%         a1 = fc_layer(layer1);
%         [loss1, ~] = svm_loss(a1, y);
%         
%         layer2 = layer;
%         layer2.b(i) = layer2.b(i) - h;
%         a2 = fc_layer(layer2);
%         [loss2, ~] = svm_loss(a2, y);
%         
%         dW_numerical = (loss1 - loss2) / (2 * h);
%         dW_target = dW(i);
%         rel_error = abs(dW_numerical - dW_target) / ...
%                       (abs(dW_numerical) + abs(dW_target));
%         fprintf('#%d, numerical: %f analytic: %f, relative error: %f\n', ...
%                       i, dW_numerical, dW_target, rel_error);
%         
%     end
% 
% end