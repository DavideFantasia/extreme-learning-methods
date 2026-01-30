function [v] = Householder_vector(x)
    % Calcola il vettore di Householder v
    % tale che v = x - s * e1
    
    % Check che x sia un vettore colonna
    x = x(:);
    n = length(x);

    %sigma = sqrt(sum(x.^2)); 
    sigma = norm(x);
    if sigma == 0
        s = 0;
        v=zeros(n,1); %vettore nullo
        return;
    end

    % Scelta di s: opposto al segno di x(1) per stabilità
    if x(1) >= 0
        s = -sigma;
    else
        s = sigma;
    end

    % Calcolo del vettore v = x - s * e1
    v = x;
    v(1) = v(1) - s;
    
    v_norm = norm(v);
    if v_norm == 0 %caso raro, implica alpha = 0
        v = zeros(length(x),1);
    else %caso con alpha=2
        v=v/norm(v); %normalizzazione per ottenere vettore unitario
    end
end