function model = elm_train(X, Y, hidden_dim, lambda, activation)
    
    N = size(X,1);
    input_dim = size(X,2);
    output_dim = size(Y,2);
    
    % Random hidden layer
    W1 = randn(hidden_dim, input_dim);
    b1 = randn(hidden_dim, 1);
    
    % Compute design matrix H = σ(W1 X + b1)
    H = activation(W1*X' + b1);   % hidden_dim × N
    H = H';                       % N × hidden_dim
    
    % Solve regularized least squares via thin QR
    [R, Vs] = thin_qr([H; sqrt(lambda)*eye(hidden_dim)]);
    
    % Augmented target
    Y_aug = [Y; zeros(hidden_dim, output_dim)];
    
    % Apply Q^T to Y_aug
    Y_tilde = zeros(size(Y_aug));
    for j = 1:output_dim
        Y_tilde(:,j) = apply_Qt(Vs, Y_aug(:,j));
    end
    
    % Final solution: R * W2 = Y_tilde(1:hidden_dim,:)
    W2 = R \ Y_tilde(1:hidden_dim,:);
    
    model.W1 = W1;
    model.b1 = b1;
    model.W2 = W2;
    model.activation = activation;
end
