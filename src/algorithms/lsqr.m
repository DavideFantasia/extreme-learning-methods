function [x, residuals, iter] = lsqr(A, b, max_iter, tol)
% LSQR - Paige & Saunders (1982)
% Risolve min ||Ax - b||²

[~, n] = size(A);

if nargin < 3, max_iter = min(size(A,1), n) + 100; end
if nargin < 4, tol = 1e-6; end

x         = zeros(n, 1);
residuals = zeros(max_iter, 1);

beta  = norm(b);
u     = b / beta;
v     = A' * u;
alpha = norm(v);
v     = v / alpha;
w     = v;

phi_bar = beta;
rho_bar = alpha;
normb   = beta;

for iter = 1:max_iter

    % Bidiagonalizzazione
    u     = A*v - alpha*u;   beta  = norm(u);  u = u/beta;
    v     = A'*u - beta*v;   alpha = norm(v);  v = v/alpha;

    % Rotazione di Givens
    rho     = sqrt(rho_bar^2 + beta^2);
    c       = rho_bar / rho;
    s       = beta    / rho;
    theta   = s * alpha;
    rho_bar = -c * alpha;
    phi     = c * phi_bar;
    phi_bar = s * phi_bar;

    % Aggiorna x e w
    x = x + (phi/rho) * w;
    w = v - (theta/rho) * w;

    % Residuo e criterio di stop (identico a CG: ||r|| < tol)
    normr = abs(phi_bar);
    residuals(iter) = normr;

    if normr < tol * normb
        break
    end
end

residuals = residuals(1:iter);
end