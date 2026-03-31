function [x, residuals, iter] = conjugate_gradient(A, b, max_iter, tol)
% conjugate_gradient risolve il sistema lineare A*x=b con il metodo del CG
% il metodo è applicabile solo se A è SPD (Symmetric Positive Definite)
% input:
% - A : matrice SPD n x n
% - b : vettore termine noto n x 1
% - max_iter: numero massimo di iterazioni consentite
% - tol : tolleranza sul residuo per il criterio di arresto
%
% output:
% - x  : soluzione approssimata al momento dell'arresto
% - residuals: vettore delle norme del residuo ad ogni iterazione
% - iter : numero di iterazioni effettivamente eseguite

    n = length(b);

    % inizializzazione
    x = zeros(n,1); % punto di partenza
    r = b - A*x; % residuo iniziale
    p = r; % prima direzione di ricerca
    rsold = dot(r,r); % ||r_0||^2, usato sia nel criterio di arresto che per beta
    
    % preallocazione del vettore dei residui alla dimensione massima
    % possibile; verrà troncato poi in base al numero di iterazioni
    % effettive
    residuals = zeros(max_iter,1);

    % confronto sul quadrato della nomra per evitare una radice quadrata ad
    % ogni iterazione
    tol2 = tol^2;
    
    % loop principale
    % ad ogni iterazione k:
    % 1. si calcola il prodotto A*p_k
    % 2. si sceglie alpha_k che minimizza f lungo la direzione p_k
    % 3. si aggiornano x, r e si verifica la convergenza
    % 4. si costruisce la nuova direzione p_{k+1} coniugata a tutte le
    % precedenti
    for iter = 1:max_iter

        Ap = A*p; % O(n^2)
        alpha = rsold / dot(p,Ap); % passo ottimale lungo la direzione p_k
        
        x = x + alpha*p;  % aggiornamento soluzione
        r = r - alpha*Ap; % aggiornamento residuo
        
        rsnew = dot(r,r); %||r_{k+1}||^2
        residuals(iter) = sqrt(rsnew); % salva ||r_{k+1}||^2 per la storia
        
        % criterio di arresto
        if rsnew < tol2 
            break;
        end
        
        p = r + (rsnew/rsold)*p; % Gram-Schimdt coniugato
        rsold = rsnew; % aggiornamento ||r_{k+1}||^2 per l'iterazione successiva
    end
    
    residuals = residuals(1:iter); % troncamento vettore dei residui
end
