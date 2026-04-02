% ########################################################################
%Vengono effettuati i seguenti test/misurazioni:
% - Tempo di training con numero di samples fissato all aumentare della
%   dimensione del layer nascosto
% - Errore nel training all aumentare del numero di neuroni con numero d
%   samples fissato
% - per CG: numero di iterazioni all aumentare della dimensione del layer 
%   nascosto, con numero di samples fissato
% ########################################################################

% Pulizia dell ambiente
clear; clc; close all;
rng(0);

% Aggiunta sottocartelle al path
addpath(genpath('algorithms'));
addpath(genpath('utils'));

reevaluation_number = 10;

% algoritmo scelto per la risoluzione del sistema lineare: 'cg', 'lsqr', o 'qr'
method = 'qr';
n_samples = [100 200 500 1000 2000];
n = n_samples(4);

% Fake dataset
X = rand(n, 2);    % n samples, m features
Y = sin(X(:,1)) + X(:,2).^2; % funzione obbiettivo

%split 80/20 del dataset in dataset di training e di test
idx = randperm(n);
ntrain = round(0.8*n);

X_train = X(idx(1:ntrain),:);
Y_train = Y(idx(1:ntrain));

X_test = X(idx(ntrain+1:end),:);
Y_test = Y(idx(ntrain+1:end));

% Hyperparameters
lambda = 1e-3; % regularization term
hidden_values = [100 150 200 250 300 350 400 500 600 700]; %numero di neuroni nel layer nascosto
activation = @(z) tanh(z);
    
% Misurazioni
avg_train_error = zeros(length(hidden_values),1);
avg_test_error  = zeros(length(hidden_values),1);
avg_times       = zeros(length(hidden_values),1);
avg_iters       = zeros(length(hidden_values),1);

% Batteria di test
for i = 1:length(hidden_values)
    cumulative.times = 0;
    cumulative.iterations = 0;
    cumulative.test_error = 0;
    cumulative.train_error = 0;
    
    hidden_dim = hidden_values(i);
    fprintf("test #%d, hidden layer dimension: %d\n",i,hidden_dim);

    for j = 1:reevaluation_number
        fprintf("\tReEvaluation number: %d/%d\n",j,reevaluation_number);
        model = elm_train(X_train, Y_train, hidden_dim, lambda, activation, method);

        % Train prediction
        Yhat_train = elm_predict(model, X_train);
        % Test prediction
        Yhat_test = elm_predict(model, X_test);
        
        %misurazioni cumulative per poter calcolare la media dopo
        cumulative.times = cumulative.times + model.solving_time;
        cumulative.iterations = cumulative.iterations + model.iteration;
        cumulative.test_error = cumulative.test_error + norm(Y_test - Yhat_test)/sqrt(length(Y_test));
        cumulative.train_error = cumulative.train_error + (norm(Y_train - Yhat_train)/sqrt(length(Y_train)));
    end

    avg_times(i) = cumulative.times/reevaluation_number;
    avg_iters(i) = cumulative.iterations/reevaluation_number;

    avg_train_error(i) = cumulative.train_error/reevaluation_number;
    avg_test_error(i) = cumulative.test_error/reevaluation_number;

end

% printing dei grafici
% Errore medio nel training all aumentare del numero di neuroni
figure
plot(hidden_values, avg_test_error,'-o','LineWidth',2)
title(strcat(method,':Average test Error on Hidden Dimension'))
xlabel('Hidden dimension')
ylabel('Test RMSE')
grid on

% Tempo medio impiegato nel training all aumentare del numero di neuroni
figure
plot(hidden_values, avg_times,'-o','LineWidth',2)
title(strcat(method,':Average training time on Hidden Dimension'))
xlabel('Hidden dimension')
ylabel('Solve time (s)')
grid on

if strcmpi(method, 'cg')
    figure
    plot(hidden_values, avg_iters,'-o','LineWidth',2)
    title(strcat(method,':Average number of iteration on Hidden Dimension'))
    xlabel('Hidden dimension')
    ylabel('CG iterations')
    grid on
end