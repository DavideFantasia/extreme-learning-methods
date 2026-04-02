% #######################################################################
% script di confronto tra algoritmi di training per ELM
% confronto dei tre metodi visti:
% - CG
% - LSQR
% - QR
% per ogni metodo e ogni dimensione dell hidden layer vengono misurati:
% - tempo medio di training
% - numero medio di iterazioni (per CG e LSQR)
% - RMSE medio sul test set
% i risulat vengono visualizzati trmite dei plot e salvati nella cartella
% results/ come PNG
% #######################################################################

clear; clc; close all;
rng(0); % seed fisso per riproducibilità dei risultati

% path per cartelle con algo e utility
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'algorithms'));
addpath(fullfile(script_dir, 'utils'));


% generazione dataset sintetico
n = 1000;
X = rand(n, 2); %n=1000 campioni, d=2 feature
Y = sin(X(:,1)) + X(:,2).^2; % funzione target non lineare

% split train/test (80%,20%) con permutazione casuale
idx    = randperm(n);
ntrain = round(0.8 * n);    % 800 campioni per il training

X_train = X(idx(1:ntrain), :);
Y_train = Y(idx(1:ntrain));
X_test  = X(idx(ntrain+1:end), :);
Y_test  = Y(idx(ntrain+1:end));

% configurazione iperparametri
lambda              = 1e-3;
activation          = @(z) tanh(z);
hidden_values       = [20 50 100 150 200 250 300 350 400 500 600 700];
reevaluation_number = 10;
nH                  = length(hidden_values);

% preallocazione dei vettori per i risultati aggregati
avg_times_cg = zeros(nH, 1);  % tempi medi CG
avg_iters_cg = zeros(nH, 1);  % iterazioni medie CG
avg_err_cg   = zeros(nH, 1);    % RMSE medio CG

avg_times_lsqr  = zeros(nH, 1);  % tempi medi LSQR
avg_iters_lsqr  = zeros(nH, 1);  % iterazioni medie LSQR
avg_err_lsqr = zeros(nH, 1);    % RMSE medio LSQR

avg_times_qr = zeros(nH, 1);  % tempi medi QR
avg_err_qr = zeros(nH, 1); % RMSE medio QR


% batteria di test principale
% - loop esterno: varia la dimensione del layer nascosto m
% - loop interno: ripete l esperimento reevolution_number volte per stimare
% medie più robuste
for i = 1:nH    % loop esterno
    hidden_dim = hidden_values(i);
    fprintf('Test #%d -- hidden_dim = %d\n', i, hidden_dim);

    cum_time_cg   = 0; cum_iter_cg   = 0; cum_err_cg   = 0;
    cum_time_lsqr = 0; cum_iter_lsqr = 0; cum_err_lsqr = 0;
    cum_time_qr   = 0; cum_err_qr    = 0;

    for j = 1:reevaluation_number % loop interno
        fprintf('\tRipetizione %d/%d\n', j, reevaluation_number);

        % ---- CG ----
        % elm_train restituisce un modello con campi: W1, b, W2,
        % solving_time (tempo di risoluzione del sistema lineare),
        % iteration (numero di iterazioni CG a convergenza).
        model_cg    = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'cg');
        Yhat_cg     = elm_predict(model_cg, X_test);
        cum_time_cg = cum_time_cg + model_cg.solving_time;
        cum_iter_cg = cum_iter_cg + model_cg.iteration;
        cum_err_cg  = cum_err_cg  + norm(Y_test - Yhat_cg) / sqrt(length(Y_test));

        % ---- LSQR ----
        model_lsqr    = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'lsqr');
        Yhat_lsqr     = elm_predict(model_lsqr, X_test);
        cum_time_lsqr = cum_time_lsqr + model_lsqr.solving_time;
        cum_iter_lsqr = cum_iter_lsqr + model_lsqr.iteration;
        cum_err_lsqr  = cum_err_lsqr  + norm(Y_test - Yhat_lsqr) / sqrt(length(Y_test));

        % QR
        % non ha campo iteration
        model_qr    = elm_train(X_train, Y_train, hidden_dim, lambda, activation, 'qr');
        Yhat_qr     = elm_predict(model_qr, X_test);
        cum_time_qr = cum_time_qr + model_qr.solving_time;
        cum_err_qr  = cum_err_qr  + norm(Y_test - Yhat_qr) / sqrt(length(Y_test));
    end

    % calcolo delle medie su reevaluation_number ripetizioni
    avg_times_cg(i)   = cum_time_cg   / reevaluation_number;
    avg_iters_cg(i)   = cum_iter_cg   / reevaluation_number;
    avg_err_cg(i)     = cum_err_cg    / reevaluation_number;

    avg_times_lsqr(i) = cum_time_lsqr / reevaluation_number;
    avg_iters_lsqr(i) = cum_iter_lsqr / reevaluation_number;
    avg_err_lsqr(i)   = cum_err_lsqr  / reevaluation_number;

    avg_times_qr(i)   = cum_time_qr   / reevaluation_number;
    avg_err_qr(i)     = cum_err_qr    / reevaluation_number;
end

% ========================================
% Figura 1 -- Iterazioni CG vs hidden_dim
% ========================================
figure;
plot(hidden_values, avg_iters_cg, '-s', 'LineWidth', 2, 'MarkerSize', 7, ...
     'Color', [0.85 0.33 0.10]);
title('cg: Average CG Iterations vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Average CG Iterations');
grid on;

% =========================================
% Figura 2 -- Iterazioni LSQR vs hidden_dim 
% =========================================
figure;
plot(hidden_values, avg_iters_lsqr, '-o', 'LineWidth', 2, 'MarkerSize', 7, ...
     'Color', [0.00 0.45 0.74]);
title('lsqr: Average LSQR Iterations vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Average LSQR Iterations');
grid on;

% =============================================
% Figura 3 -- Iterazioni CG vs LSQR a confronto
% =============================================
figure;
plot(hidden_values, avg_iters_cg,   '-s', 'LineWidth', 2, 'MarkerSize', 7, ...
     'Color', [0.85 0.33 0.10], 'DisplayName', 'CG');
hold on;
plot(hidden_values, avg_iters_lsqr, '-o', 'LineWidth', 2, 'MarkerSize', 7, ...
     'Color', [0.00 0.45 0.74], 'DisplayName', 'LSQR');
title('CG vs LSQR: Average Iterations vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Average Iterations');
legend('Location', 'northwest');
grid on;

% ==============================================
% Figura 4 -- Tempo di training CG vs LSQR vs QR
% ==============================================
figure;
plot(hidden_values, avg_times_cg,   '-o', 'LineWidth', 2, 'DisplayName', 'CG');
hold on;
plot(hidden_values, avg_times_lsqr, '-s', 'LineWidth', 2, 'DisplayName', 'LSQR');
plot(hidden_values, avg_times_qr,   '-^', 'LineWidth', 2, 'DisplayName', 'QR');
title('CG vs LSQR vs QR: Average Training Time vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Solve Time (s)');
legend('Location', 'northwest');
grid on;

% ==========================================
% Figura 5 -- RMSE test set CG vs LSQR vs QR
% ==========================================
figure;
plot(hidden_values, avg_err_cg,   '-o', 'LineWidth', 2, 'DisplayName', 'CG');
hold on;
plot(hidden_values, avg_err_lsqr, '-s', 'LineWidth', 2, 'DisplayName', 'LSQR');
plot(hidden_values, avg_err_qr,   '-^', 'LineWidth', 2, 'DisplayName', 'QR');
title('CG vs LSQR vs QR: Average Test RMSE vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Test RMSE');
legend('Location', 'northeast');
grid on;

% =============================================================
% Figura 6 -- Log-log scalabilità CG e LSQR con stima esponente
% =============================================================
figure;
loglog(hidden_values, avg_times_cg,   '-o', 'LineWidth', 2, 'DisplayName', 'CG');
hold on;
loglog(hidden_values, avg_times_lsqr, '-s', 'LineWidth', 2, 'DisplayName', 'LSQR');

p_cg   = polyfit(log(hidden_values), log(avg_times_cg'),   1);
p_lsqr = polyfit(log(hidden_values), log(avg_times_lsqr'), 1);
loglog(hidden_values, exp(p_cg(2))   * hidden_values .^ p_cg(1),   '--', ...
       'LineWidth', 1.5, 'Color', [0.85 0.33 0.10], ...
       'DisplayName', sprintf('CG fit (p=%.2f)',   p_cg(1)));
loglog(hidden_values, exp(p_lsqr(2)) * hidden_values .^ p_lsqr(1), '--', ...
       'LineWidth', 1.5, 'Color', [0.00 0.45 0.74], ...
       'DisplayName', sprintf('LSQR fit (p=%.2f)', p_lsqr(1)));

title('CG vs LSQR: Log-Log Training Time vs Hidden Dimension');
xlabel('Hidden Dimension m');
ylabel('Solve Time (s)');
legend('Location', 'northwest');
grid on;

fprintf('\nStima esponente scalabilità CG:   p = %.2f\n', p_cg(1));
fprintf('Stima esponente scalabilità LSQR: p = %.2f\n', p_lsqr(1));

% ===================
% salvataggio figure
% ===================
outdir = fullfile(script_dir, 'results');
if ~exist(outdir, 'dir'), mkdir(outdir); end

figHandles = findall(groot, 'Type', 'figure');
for k = 1:numel(figHandles)
    fh = figHandles(k);
    drawnow;
    filename = sprintf('%02d_figure_%d.png', k, fh.Number);
    exportgraphics(fh, fullfile(outdir, filename), 'Resolution', 300);
end