% test_lsqr.m
% Analisi delle prestazioni di LSQR:
%   - Numero di iterazioni LSQR vs dimensione layer nascosto
%   - Tempo di training LSQR vs QR vs dimensione layer nascosto
%   - RMSE sul test set LSQR vs QR vs dimensione layer nascosto

clear; clc; close all;
rng(0);

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'algorithms'));
addpath(fullfile(script_dir, 'utils'));

% =========================================================
% Dataset sintetico (identico a test_cg.m)
% =========================================================
n = 1000;
X = rand(n, 2);
Y = sin(X(:,1)) + X(:,2).^2;

idx    = randperm(n);
ntrain = round(0.8 * n);

X_train = X(idx(1:ntrain), :);
Y_train = Y(idx(1:ntrain));
X_test  = X(idx(ntrain+1:end), :);
Y_test  = Y(idx(ntrain+1:end));

% =========================================================
% Hyperparameters
% =========================================================
lambda        = 1e-3;
activation    = @(z) tanh(z);
hidden_values = [20 50 100 150 200 250 300 350 400 500 600 700];
reevaluation_number = 10;

nH = length(hidden_values);

avg_times_lsqr    = zeros(nH, 1);
avg_iters_lsqr    = zeros(nH, 1);
avg_test_err_lsqr = zeros(nH, 1);
avg_times_qr      = zeros(nH, 1);
avg_test_err_qr   = zeros(nH, 1);

% =========================================================
% Batteria di test
% =========================================================
for i = 1:nH
    hidden_dim = hidden_values(i);
    fprintf('Test #%d -- hidden_dim = %d\n', i, hidden_dim);

    cum_time_lsqr = 0; cum_iter_lsqr = 0; cum_err_lsqr = 0;
    cum_time_qr   = 0; cum_err_qr    = 0;

    for j = 1:reevaluation_number
        fprintf('\tRipetizione %d/%d\n', j, reevaluation_number);

        % LSQR
        model_lsqr     = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'lsqr');
        Yhat_test_lsqr = elm_predict(model_lsqr, X_test);

        cum_time_lsqr = cum_time_lsqr + model_lsqr.solving_time;
        cum_iter_lsqr = cum_iter_lsqr + model_lsqr.iteration;
        cum_err_lsqr  = cum_err_lsqr  + norm(Y_test - Yhat_test_lsqr) / sqrt(length(Y_test));

        % QR
        model_qr     = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'qr');
        Yhat_test_qr = elm_predict(model_qr, X_test);

        cum_time_qr = cum_time_qr + model_qr.solving_time;
        cum_err_qr  = cum_err_qr  + norm(Y_test - Yhat_test_qr) / sqrt(length(Y_test));
    end

    avg_times_lsqr(i)    = cum_time_lsqr / reevaluation_number;
    avg_iters_lsqr(i)    = cum_iter_lsqr / reevaluation_number;
    avg_test_err_lsqr(i) = cum_err_lsqr  / reevaluation_number;
    avg_times_qr(i)      = cum_time_qr   / reevaluation_number;
    avg_test_err_qr(i)   = cum_err_qr    / reevaluation_number;
end

% =========================================================
% Figura 1 -- Iterazioni LSQR vs hidden_dim
% =========================================================
figure;
plot(hidden_values, avg_iters_lsqr, '-s', 'LineWidth', 2, 'MarkerSize', 7, ...
     'Color', [0.00 0.45 0.74]);
title('lsqr: Average LSQR Iterations vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Average LSQR Iterations');
grid on;

% =========================================================
% Figura 2 -- Tempo di training LSQR vs QR
% =========================================================
figure;
plot(hidden_values, avg_times_lsqr, '-o', 'LineWidth', 2, 'DisplayName', 'LSQR');
hold on;
plot(hidden_values, avg_times_qr,   '-s', 'LineWidth', 2, 'DisplayName', 'QR');
title('LSQR vs QR: Average Training Time vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Solve Time (s)');
legend('Location', 'best');
grid on;

% =========================================================
% Figura 3 -- RMSE test set LSQR vs QR
% =========================================================
figure;
plot(hidden_values, avg_test_err_lsqr, '-o', 'LineWidth', 2, 'DisplayName', 'LSQR');
hold on;
plot(hidden_values, avg_test_err_qr,   '-s', 'LineWidth', 2, 'DisplayName', 'QR');
title('LSQR vs QR: Average Test RMSE vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Test RMSE');
legend('Location', 'best');
grid on;

% =========================================================
% Figura 4 -- Log-log scalabilità LSQR con stima esponente
% =========================================================
figure;
loglog(hidden_values, avg_times_lsqr, '-o', 'LineWidth', 2, 'DisplayName', 'LSQR time');
hold on;
p_lsqr = polyfit(log(hidden_values), log(avg_times_lsqr), 1);
y_fit  = exp(p_lsqr(2)) * hidden_values .^ p_lsqr(1);
loglog(hidden_values, y_fit, '--r', 'LineWidth', 2, 'DisplayName', 'Linear fit');
title('lsqr: Log-Log Training Time vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Solve Time (s)');
legend('Location', 'best');
grid on;
fprintf('\nStima esponente scalabilità LSQR: p = %.2f\n', p_lsqr(1));

% =========================================================
% Salvataggio figure
% =========================================================
outdir = fullfile(script_dir, 'results');
if ~exist(outdir, 'dir'), mkdir(outdir); end

figHandles = findall(groot, 'Type', 'figure');
for k = 1:numel(figHandles)
    fh = figHandles(k);
    drawnow;
    filename = sprintf('%02d_figure_%d.png', k, fh.Number);
    exportgraphics(fh, fullfile(outdir, filename), 'Resolution', 300);
end