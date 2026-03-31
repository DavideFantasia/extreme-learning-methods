% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% script per analizzare le prestazioni del metodo (A2) CG applicato al
% training dell'ELM
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\

clear; clc; close all;
rng(0); % seed fisso garantisce riproducibilità

% path per cartelle con algo e utility
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'algorithms'));
addpath(fullfile(script_dir, 'utils'));

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% generazione dataset sintetico
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
n = 1000;
X = rand(n, 2); % n=1000 campioni, d=2 feature
Y = sin(X(:,1)) + X(:,2).^2; % funzione target non lineare

% split train/test (80%,20%) con permutazione casuale
idx    = randperm(n);
ntrain = round(0.8 * n); % 800 campioni per il training

X_train = X(idx(1:ntrain), :);
Y_train = Y(idx(1:ntrain));
X_test  = X(idx(ntrain+1:end), :);
Y_test  = Y(idx(ntrain+1:end));

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% configurazione iperparametri
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
lambda        = 1e-3;
activation    = @(z) tanh(z);
hidden_values = [20 50 100 150 200 250 300 350 400 500 600 700];
reevaluation_number = 10;

nH = length(hidden_values);

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% preallocazione dei vettori per i risultati aggregati
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
avg_times_cg    = zeros(nH, 1);
avg_iters_cg    = zeros(nH, 1);
avg_test_err_cg = zeros(nH, 1);
avg_times_qr    = zeros(nH, 1);
avg_test_err_qr = zeros(nH, 1);

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% batteria di test principale
% - loop esterno: varia la dimensione del layer nascosto m
% - loop interno: ripete l'esperimento reevolution_number volte per stimare
% medie più robuste
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
for i = 1:nH % loop esterno
    hidden_dim = hidden_values(i);
    fprintf('Test #%d -- hidden_dim = %d\n', i, hidden_dim);

    cum_time_cg = 0;  cum_iter_cg = 0;  cum_err_cg = 0;
    cum_time_qr = 0;  cum_err_qr  = 0;

    for j = 1:reevaluation_number % loop interno
        fprintf('\tRipetizione %d/%d\n', j, reevaluation_number);

        % CG
        model_cg     = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'cg');
        Yhat_test_cg = elm_predict(model_cg, X_test);

        cum_time_cg = cum_time_cg + model_cg.solving_time;
        cum_iter_cg = cum_iter_cg + model_cg.iteration;
        cum_err_cg  = cum_err_cg  + norm(Y_test - Yhat_test_cg) / sqrt(length(Y_test));

        % QR (per confronto)
        model_qr     = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'qr');
        Yhat_test_qr = elm_predict(model_qr, X_test);

        cum_time_qr = cum_time_qr + model_qr.solving_time;
        cum_err_qr  = cum_err_qr  + norm(Y_test - Yhat_test_qr) / sqrt(length(Y_test));
    end

    % --- media delle metriche accumulate ---
    avg_times_cg(i)    = cum_time_cg / reevaluation_number;
    avg_iters_cg(i)    = cum_iter_cg / reevaluation_number;
    avg_test_err_cg(i) = cum_err_cg  / reevaluation_number;
    avg_times_qr(i)    = cum_time_qr / reevaluation_number;
    avg_test_err_qr(i) = cum_err_qr  / reevaluation_number;
end

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% Figura 1 -- Iterazioni CG vs hidden_dim
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
figure;
plot(hidden_values, avg_iters_cg, '-s', 'LineWidth', 2, 'MarkerSize', 7, ...
     'Color', [0.85 0.33 0.10]);
title('cg: Average CG Iterations vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Average CG Iterations');
grid on;

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% Figura 2 -- Tempo di training CG vs QR
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
figure;
plot(hidden_values, avg_times_cg, '-o', 'LineWidth', 2, 'DisplayName', 'CG');
hold on;
plot(hidden_values, avg_times_qr, '-s', 'LineWidth', 2, 'DisplayName', 'QR');
title('CG vs QR: Average Training Time vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Solve Time (s)');
legend('Location', 'best');
grid on;

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% Figura 3 -- RMSE test set CG vs QR
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
figure;
plot(hidden_values, avg_test_err_cg, '-o', 'LineWidth', 2, 'DisplayName', 'CG');
hold on;
plot(hidden_values, avg_test_err_qr, '-s', 'LineWidth', 2, 'DisplayName', 'QR');
title('CG vs QR: Average Test RMSE vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Test RMSE');
legend('Location', 'best');
grid on;

% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
% Figura 4 -- Log-log scalabilita CG con stima esponente
% /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
figure;
loglog(hidden_values, avg_times_cg, '-o', 'LineWidth', 2, 'DisplayName', 'CG time');
hold on;
p_cg  = polyfit(log(hidden_values), log(avg_times_cg), 1);
y_fit = exp(p_cg(2)) * hidden_values .^ p_cg(1);
loglog(hidden_values, y_fit, '--r', 'LineWidth', 2, 'DisplayName', 'Linear fit');
title('cg: Log-Log Training Time vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Solve Time (s)');
legend('Location', 'best');
grid on;
fprintf('\nStima esponente scalabilita CG: p = %.2f\n', p_cg(1));

% salvataggio automatico delle figure generate sopra
outdir = fullfile(script_dir, 'results');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

figHandles = findall(groot, 'Type', 'figure');
for k = 1:numel(figHandles)
    fh = figHandles(k);
    drawnow;
    filename = sprintf('%02d_figure_%d.png', k, fh.Number);
    filepath = fullfile(outdir, filename);
    exportgraphics(fh, filepath, 'Resolution', 300);
end