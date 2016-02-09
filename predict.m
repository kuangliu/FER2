function predict(W, X, y)
%PREDICT the accuracy of trained W
    scores = X * W;  
    [~, y_pred] = max(scores, [], 2);
    
    num_correct = sum(y_pred==y);
    fprintf('correct/total = %d/%d = %f\n', num_correct, size(X,1), num_correct/size(X,1));
    
end

