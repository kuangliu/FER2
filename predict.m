function predict(W, X, y)
%PREDICT the accuracy of trained W
    scores = W*X;  
    [~, y_pred] = max(scores);
    
    num_correct = sum(y_pred==y);
    fprintf('correct/total = %d/%d = %f\n', num_correct, size(X,1), num_correct/size(X,1));
    
end

