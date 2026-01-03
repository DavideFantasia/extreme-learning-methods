function y = apply_Qt(Vs, b)
    % Applica Q^T * b, dove Q = H1 H2 ... Hn
    n = length(Vs);
    y = b;
    for k = 1:n
        v = Vs{k};
        y(k:end) = y(k:end) - 2 * v * (v' * y(k:end));
    end
end

