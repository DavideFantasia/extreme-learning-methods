function model = elm_train(X, Y, hidden_dim, lambda, activation, method)
    % method: 'qr' (default) oppure 'cg'
    
    if nargin < 6
        method = 'qr';
    end
    % Si centra il dataset sullo zero per sfruttare meglio la zona lineare
    % della funzione di attivazione, evitando eventuali saturazioni
    model.mu = mean(X);
    model.sigma = std(X) + 1e-8;
    X = (X - model.mu) ./ model.sigma; 

    N = size(X,1);
    input_dim = size(X,2);
    output_dim = size(Y,2);
    
    % Random hidden layer
    W1 = randn(hidden_dim, input_dim)/ sqrt(input_dim); %con scalatura per input_dim
    b1 = randn(hidden_dim, 1);
    
    %H = activation(W1*X' + b1);   % hidden_dim x N
    %H = H';                       % N x hidden_dim
    %evitiamo il calcolo esplicito come:
    H = activation(X*W1' + b1');

    % Soluzione per W2 basata su 'method'
    if strcmpi(method, 'qr')
        % --- Metodo Thin QR ---
        % Risolve: min || [H; sqrt(lambda)I] * W2 - [Y; 0] ||^2
        
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
        
    elseif strcmpi(method, 'cg')
        % --- Metodo Conjugate Gradient ---
        % Risolve il sistema lineare: (H'H + lambda*I) * W2 = H'Y
        
        % Costruzione della matrice del sistema A (simmetrica e definita positiva)
        A = (H' * H) + lambda * eye(hidden_dim);
        
        % Costruzione del termine noto B
        B_target = H' * Y;
        
        % Inizializzazione pesi
        W2 = zeros(hidden_dim, output_dim);
        
        % Parametri CG
        max_iter = 1000; % O hidden_dim
        tol = 1e-6; % tolleranza
        
        % Il CG risolve Ax=b per un vettore b. 
        % Poiché Y può avere più colonne (multi-output), cicliamo su ogni output.
        for j = 1:output_dim
            b_j = B_target(:, j);
            
            % Chiamata alla funzione conjugate_gradient fornita
            [W2(:, j), res, iter] = conjugate_gradient(A, b_j, max_iter, tol);
            
        end
        
    else
        error('Metodo non riconosciuto. Usa "qr" o "cg".');
    end
    
    % save model data
    model.W1 = W1;
    model.b1 = b1;
    model.W2 = W2;
    model.activation = activation;
    model.method = method;
end