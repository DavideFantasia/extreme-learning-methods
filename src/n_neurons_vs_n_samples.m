%Si vuole testare se, con la soluzione tramite QR, aumentando il numero di samples
% e lasciando fissa hidden_dim, ci si aspetta complessità lineare in tempo

% Pulizia dell'ambiente
clear; clc; close all;

% Aggiunta sottocartelle al path
addpath(genpath('algorithms'));
addpath(genpath('utils'));
rng(0); %seed del random, per fissare la generazione di W1

% algoritmo scelto per la risoluzione del sistema lineare: 'cg' o 'qr'
method = 'qr';
n_samples = [2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000];

% Hyperparameters
lambda = 1e-3; % regularization term
hidden_values = [20 50 100 150 200 250 300 350 400 500 600 700 800]; %numero di neuroni nel layer nascosto
activation = @(z) tanh(z);

% Misurazioni
avg_times       = zeros(length(n_samples),1);

reevaluation_number = 15; %number of time we compute the training to average the performance.

%dataset di dimensione massima
n_max = max(n_samples);
X_full = rand(n_max,2);
Y_full = sin(X_full(:,1)) + X_full(:,2).^2;

% Batteria di test
for i = 1:length(n_samples)
    cumulative_times = 0;
    n = n_samples(i);
    
    % subset of the fake dataset
    X = X_full(1:n,:);
    Y = Y_full(1:n);
    
    %split 80/20 del dataset in dataset di training e di test
    idx = randperm(n);
    ntrain = round(0.8*n);
    
    X_train = X(idx(1:ntrain),:);
    Y_train = Y(idx(1:ntrain));
    
    X_test = X(idx(ntrain+1:end),:);
    Y_test = Y(idx(ntrain+1:end));

    hidden_dim = hidden_values(1);

    fprintf("test #%d, number of samples: %d\n",i,n);
    for j = 1:reevaluation_number
        fprintf("\tReEvaluation number: %d/%d\n",j,reevaluation_number);

        model = elm_train(X_train, Y_train, hidden_dim, lambda, activation, method);
        cumulative_times = cumulative_times + model.solving_time;
    end
    % Train prediction
    Yhat_train = elm_predict(model, X_train);

    avg_times(i) = cumulative_times/reevaluation_number;

    % Test prediction
    Yhat_test = elm_predict(model, X_test);
end

% printing del grafico:
% tempo impiegato nel training aumentando il numero di sampling e fissando
% il numero di neuroni
figure
loglog(n_samples, avg_times,'-o','LineWidth',2)
% Fit lineare: y = a*x + b
p = polyfit(log(n_samples), log(avg_times), 1);
% Valori stimati sulla stessa ascissa
y_fit = exp(p(2)) * n_samples.^p(1);
hold on
loglog(n_samples, y_fit, '--r', 'LineWidth', 2)
legend('Measured time','Linear fit','Location','best')
title(strcat(method,':Time Elapsed on training based on Samples Number'))
xlabel('Samples Number')
ylabel('Time Elapsed')
grid on

disp(['Estimated order: ', num2str(p(1))])
