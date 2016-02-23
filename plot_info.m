function plot_info(info, x)
%PLOT_INFO plot the information of each training epoch including:
%   - train_losses
%   - val_losses
%   - val_accuracies

subplot(1,2,1)
plot(1:x, info.train_losses(1:x), 1:x, info.val_losses(1:x))
title('Loss')
xlabel('epoch')
legend('train','val')

subplot(1,2,2)
plot(1:x, info.val_accuracies(1:x))
title('Accuracy')
xlabel('epoch') 
ylabel('%')
legend('val')




