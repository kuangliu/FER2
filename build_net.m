function net = build_net()
%BUILD_NET build the network
    C = 10;     % C classes
    D = 3073;   % D dimension
    
    net = {};
    
    % 1. fc layer
    fc1.type = 'fc';
    fc1.W = randn(C, D, 'single') / sqrt(D);
    net{end+1} = fc1;   % add to the network

    

end

