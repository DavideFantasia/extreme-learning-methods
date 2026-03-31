function [x, residuals, iter] = lsqr(A, b, max_iter, tol)
% LSQR risolve il problema min ||Ax - b||^2 tramite bidiagonalizzazione di
% Lanczos con Givens rotations
% a differenza del CG, LSQR opera direttamente su A rettangolare ed evita
% di formare esplicitamente A'A e non ne quadra il condizionamento

% ad ogni iterazione:
% 1. Bidiagonalizzazione di Lanczos
% 2. Givens rotations

% inizializzazione
[~, n] = size(A);

if nargin < 3, max_iter = min(size(A,1), n) + 100; end
if nargin < 4, tol = 1e-6; end

% partenza da solizione nulla
x         = zeros(n, 1);
residuals = zeros(max_iter, 1);

% inizializzazione di krylov
beta  = norm(b);
u     = b / beta;
v     = A' * u;
alpha = norm(v);
v     = v / alpha;
w     = v;

% variabili per Givens rotations
phi_bar = beta;
rho_bar = alpha;
normb   = beta;

for iter = 1:max_iter

    % bidiagonalizzazione
    u     = A*v - alpha*u;   beta  = norm(u);  u = u/beta;
    v     = A'*u - beta*v;   alpha = norm(v);  v = v/alpha;

    % rotazione di Givens
    rho     = sqrt(rho_bar^2 + beta^2);
    c       = rho_bar / rho;
    s       = beta    / rho;
    theta   = s * alpha;
    rho_bar = -c * alpha;
    phi     = c * phi_bar;
    phi_bar = s * phi_bar;

    % aggiorna x e w
    x = x + (phi/rho) * w;
    w = v - (theta/rho) * w;

    % residuo e criterio di stop
    normr = abs(phi_bar);
    residuals(iter) = normr;

    if normr < tol * normb
        break
    end
end

residuals = residuals(1:iter);
end