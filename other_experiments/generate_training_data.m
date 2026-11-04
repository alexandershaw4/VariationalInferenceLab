function [X_train, Y_train] = generate_training_data(f, x, V, num_samples)
    % Number of parameters
    num_params = length(x);

    % initial points
    f0 = f(x);
    
    % Get the size of the frequency domain output from one simulation
    sample_output = f(x);
    num_freqs = length(sample_output);
    
    % Preallocate storage for training data
    X_train = zeros(num_samples, num_params);
    Y_train = zeros(num_samples, num_freqs); % Store frequency-domain outputs
    
    for i = 1:num_samples
        % Perturb parameters by sampling from Gaussian distribution
        x_perturbed = x + sqrt(V) .* randn(size(x)); 
        
        % Simulate the model with perturbed parameters (already in frequency domain)
        y_sim = f(x_perturbed) - f0;
        
        % Store training samples
        X_train(i, :) = x_perturbed;
        Y_train(i, :) = y_sim; % Directly store frequency domain output
    end
end
