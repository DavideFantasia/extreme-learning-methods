function [x, residuals, iter] = conjugate_gradient(A, b, max_iter, tol)

n = length(b);
x = zeros(n,1);

r = b - A*x;
p = r;
rsold = dot(r,r);

residuals = zeros(max_iter,1);
tol2 = tol^2;

for iter = 1:max_iter
    Ap = A*p;
    alpha = rsold / dot(p,Ap);
    
    x = x + alpha*p;
    r = r - alpha*Ap;
    
    rsnew = dot(r,r);
    residuals(iter) = sqrt(rsnew);
    
    if rsnew < tol2
        break;
    end
    
    p = r + (rsnew/rsold)*p;
    rsold = rsnew;
end

residuals = residuals(1:iter);
end
