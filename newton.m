function [x, fvals] = newton(f, dims, x0, param)
% NEWTON solves argmin_x f(x) with Newton's method, where f: R^n -> R is 
% an at least twice differentiable function
%
%   Usage:
%       [t, energy] = newton(f, x0, param)
%
%   Input:
%       f       : A Matlab structure containing information about the
%                 differentiable function f. It must have as fields:
%           f.eval  : A function handle with the expression for f(t).
%           f.grad  : A function handle with the expression for the
%                     gradient of f w.r.t. x.
%           f.hess  : A function handle with the expression for the
%                     hessian of f w.r.t. x.
%       dims    : Vector with the dimensions of the argument to be learned
%                 (DEFAULT: size(x0))
%       x0      : Initialization of the argument to be learned.
%                 (DEFAULT: zeros(dims))
%       param     : Matlab structure with some additional parameters.
%           param.TOL       : Stop criterium. When the Newton decrement
%                             becomes less than TOL, we quit the iterative 
%                             process. 
%                             (DEFAULT: 1e-4).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the 
%                             iterative process. 
%                             (DEFAULT: 100).
%         
%   Output:
%         x         : Argument that minimizes f
%         fvals     : A 1-by-(k+1) vector with the values f(x(k-1)),
%                     where k is the iteration number
%
%   Example:
%       f.eval = @(x) norm(x,2).^2;
%       f.grad = @(x) 2.*x;
%       f.hess = @(x) 2;
%       x = newton(f, [], 100*randn(100,1));
%
%   See also:
%
%   References:
%
% Author: Rodrigo Pena
% Date: 15 Dec 2015
% Testing:

%% Parse input
% f
assert(isfield(f, 'eval') && isfield(f, 'grad') && isfield(f, 'hess'), ...
    'f doesn''t have the correct fields.');
assert(isa(f.eval, 'function_handle'), 'f.eval must be a function handle');
assert(isa(f.grad, 'function_handle'), 'f.grad must be a function handle');
assert(isa(f.hess, 'function_handle'), 'f.hess must be a function handle');

% dims
if isempty(dims)
    assert(nargin > 2, ...
        'Initial point must be provided if dims is not provided.');
    assert(~isempty(x0), ...
        'Initial point must be provided if dims is not provided.');
    dims = size(x0);
else
    assert(isa(dims, 'numeric'), 'The dimensions must be numeric');
    dims = round(dims);
end

% t0
if (nargin < 3) || isempty(x0); x0 = zeros(dims); end
assert(sum(dims ~= size(x0)) == 0, 'Initial point must have size dims');

% param
if (nargin < 4); param = []; end
if ~isfield(param, 'TOL'); param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 1000; end
assert(isnumeric(param.TOL) && sum(size(param.TOL)~=1) == 0, ...
    'param.TOL must be a number.');
assert(isnumeric(param.MAX_ITER) && sum(size(param.MAX_ITER)~=1) == 0, ...
    'param.MAX_ITER must be a number.');
param.MAX_ITER = abs(round(param.MAX_ITER)); 

%% Initialization
x = x0; 
k = 0;

alpha = 0.25;
beta = 0.5;
t = 1;

decrement = Inf;
fvals = zeros(1, param.MAX_ITER);
fvals(1) = f.eval(x);

%% Iterative steps of Newton's Method
while (decrement > param.TOL) && (k < param.MAX_ITER)
    
    k = k + 1;
    
    % Gradient
    Grad = f.grad(x);
    
    % Hessian
    Hess = f.hess(x);
    
    % Compute Newton step
    [~, p] = chol(Hess);
    if p == 0 % Hess is positive definite
        delta = - Hess \ Grad;
    else
        delta = - sign(Grad) * abs(pinv(Hess) * Grad);
    end
    
    % Compute Newton decrement
    decrement = - Grad' * delta;
    
    % Update x
    x_old = x;
    x = x_old + (t * delta);
    
    % Backtracking Line Search
    while f.eval(x) >= f.eval(x_old) + alpha * t * Grad' * delta
       t = beta * t;
       x = x_old + (t * delta);
    end    
    
    % Evaluate function at new point
    fvals(k + 1) = f.eval(x);
    
end

% Trim down the values vector.
fvals = fvals(1:k+1);

end