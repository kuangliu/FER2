function tmp()
    
loss = randi(100,10,1);
acy = randi(100,10,1);

subplot(1,2,1)
plot(loss) 
hold;
plot(acy)
title('Loss')
xlabel('epoch')
legend('train','val')

subplot(1,2,2)
plot(acy)
hold;
plot(acy)
title('Accuracy')
xlabel('epoch') 
ylabel('%')
legend('train','val')

