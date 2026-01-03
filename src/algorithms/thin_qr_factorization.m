function [Q,R] = thin_qr_factorization(A)
%THIN_QR Economy (thin) QR via Householder
%   [Q,R] = thin_qr(A)
%   Input:
%     A  m-by-n matrix (m >= n)
%   Output:
%     Q  m-by-n orthonormale (thin Q)
%     R  n-by-n upper triangular
%
% Nota:
%   - Questo implementa i riflettori H = I - 2*v*v' con v di norma 1.
%   - I vettori v sono salvati in una cell array Vs per ricostruire Q.
%   - Complessità O(m*n^2).
    
    [m,n] = size(A);
    if m < n
        error('thin_qr: richiede m >= n');
    end
    
    Vs = cell(n,1);   % memorizza i vettori di Householder (v con ||v||=1)
    for k = 1:n
        % estrai la sottocolonna su cui costruire il riflettore
        x = A(k:m, k);
        sigma = norm(x);
        if sigma == 0
            % riflettore identità
            v = zeros(length(x),1);
        else
            % scelta del segno per stabilità
            alpha = -sign(x(1)) * sigma;
            % costruisco v = x - alpha * e1
            v = x;
            v(1) = v(1) - alpha; %[pag 72 di Trefethen-Bau]: v=-sign(x_1) ||x||e_1 -x
            vnorm = norm(v);
            if vnorm == 0
                % caso raro numericamente
                v = zeros(length(x),1);
                tau = 0;
            else
                v = v / vnorm;   % normalizzo v -> ||v|| = 1
                tau = 2;         % H = I - 2 v v'
            end
        end
    
        % applica H = I - tau/1 * (v*v') con tau=2 (quindi H = I - 2*v*v')
        if tau ~= 0
            % block da aggiornare: A(k:m, k:n)
            % w = 2 * v' * A_block
            w = 2 * (v' * A(k:m, k:n));
            A(k:m, k:n) = A(k:m, k:n) - v * w;
        end
    
        % salva il vettore v (dimensione m-k+1)
        Vs{k} = v;
    
        % nota: dopo l'operazione, la colonna k contiene i coefficienti di R
        % nelle posizioni A(k,k:n). Le parti sotto la diagonale sono "modificate";
        % estrarremo R con triu(A(1:n,1:n)) alla fine.
    end
    
    % R è la parte triangolare superiore delle prime n righe/colonne di A
    R = triu(A(1:n,1:n));
    
    % costruisci thin Q (m-by-n) applicando i riflettori in ordine inverso a I_m
    Qfull = eye(m);
    for k = n:-1:1
        v = Vs{k};
        if ~all(v==0)
            Qfull(k:m, :) = Qfull(k:m, :) - 2 * v * (v' * Qfull(k:m, :));
        end
    end
    
    Q = Qfull(:,1:n);   % thin Q m-by-n
end
