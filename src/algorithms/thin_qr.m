function [R,V] = thin_qr(A)
% Calcolo della Thin QR Factorization di A, ma calcolando solo R e
% restituendo il vettore V di vettori di Householder:
% Hx= x- 2/norm(v) * v * v^T * x
    disp("--- thin QR Factorization ---");
    [m,n] = size(A);
    if m < n
        error('thin_qr: richiede m >= n');
    end

    V = cell(n,1);
    for k= 1:n 
        x = A(k:m,k);
        v = Householder_vector(x);
        A(k:m,k:n)=A(k:m,k:n)-2*v*(v.'* A(k:m,k:n)); 

        %salvataggio del vettore v
        V{k}=v;
    end

    %R è la triangolare superiore di A
    R = triu(A(1:n,1:n));
end