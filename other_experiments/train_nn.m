function net = train_nn(Y_train, X_train)
    num_features = size(Y_train, 2); % Number of frequency bins
    num_outputs = size(X_train, 2);  % Number of parameters

    layers = [
        featureInputLayer(num_features) % Input layer matching frequency size
        fullyConnectedLayer(100) % Hidden layer 1
        reluLayer
        fullyConnectedLayer(50) % Hidden layer 2
        reluLayer
        fullyConnectedLayer(num_outputs) % Output layer matching parameter dimension
        regressionLayer];

    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 1000, ...
        'MiniBatchSize', 32, ...
        'InitialLearnRate', 1e-3, ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % Train the network
    net = trainNetwork(Y_train, X_train, layers, options);
end
