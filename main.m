% net = build_net();
% [net, loss_history] = train_net(net, X, y_train, opts);h = animatedline;

h = animatedline;
for x = 1:10
    y = x + randi(3);
    addpoints(h, x, y);
    drawnow limitrate
end

