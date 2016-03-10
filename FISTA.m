function [x, Evals] = FISTA(g, f, N, x0, param)
% FISTA solves the following optimization problem with the Fast Iterative 
% Shrinkage-Thresholding Algorithm (FISTA):
% 
% (1)              argmin_x (g(x) + f(x)) = argmin_x E(x)
%
% where g: R^N -> R is a non-smooth convex function, and f: R^N -> R is a 
% differentiable convex function, with Lipschitz constant L.
%
%   Usage:
%       [x, Evals] = FISTA(g, f, N, x0, param)
%
%   Input:
%       g       : A Matlab structure containing information about the
%                 non-smooth convex function g. It must have as fields:
%           g.eval  : A function handle with the expression for g(x).
%           g.prox  : A function handle with the proximity operator of 
%                     tau*g(x), as a function of x and tau.
%                     (e.g., if g.eval = @(x) norm(x, 1), then
%                      g.prox = @(x,tau) wthresh(x,'s',tau)).
%       f       : A Matlab structure containing information about the
%                 differentiable function f. It must have as fields:
%           f.eval  : A function handle with the expression for f(x).
%           f.grad  : A function handle with the expression for the
%                     gradient of f w.r.t. x.
%           f.L     : (Optional) The Lipschitz constant of f(x). If not
%                     specified, FISTA will use backtracking line search 
%                     to estimate L.
%       N         : Length of the vector x to be learned.
%                   (Default: length(x0))
%       x0        : Initialization of the vector x to be learned. 
%                   (Default: zeros(N,1))
%       param     : Matlab structure with some additional parameters.
%           param.TOL       : Stop criterium. When ||x(k) - x(k-1)||_2, 
%                             where k is the iteration number, is less than
%                             TOL, we quit the iterative process. 
%                             (Default: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the 
%                             iterative process. 
%                             (Default: 1000).
%         
%   Output:
%       x       : A N-by-1 vector with the solution to the optimization 
%                 problem (1).
%       Evals   : A vector with the objective values E(x(k-1)), where k is 
%                 the iteration number.
%
%   Example:
%       [x, Evals] = FISTA(g, f, N, x0, param)
%          
%   See also:
%
%   References:
%       [1]	A. Beck and M. Teboulle, "A Fast Iterative Shrinkage-
%       Thresholding Algorithm for Linear Inverse Problems," SIAM J. 
%       Imaging Sciences, vol. 2, pp. 183-202, 2009.
%
% Author: Rodrigo Pena
% Date: 15 Dec 2015
% Testing:

%% Parse input
% g
assert(isfield(g, 'eval') && isfield(g, 'prox'), ...
    'g doesn''t have the correct fields.');
assert(isa(g.eval, 'function_handle'), 'g.eval must be a function handle');
assert(isa(g.prox, 'function_handle'), 'g.prox must be a function handle');

% f
assert(isfield(f, 'eval') && isfield(f, 'grad'), ...
    'f doesn''t have the correct fields.');
assert(isa(f.eval, 'function_handle'), 'f.eval must be a function handle');
assert(isa(f.grad, 'function_handle'), 'f.grad must be a function handle');
if ~isfield(f, 'L') || isempty(f.L)
    backtracking_flag = 1;
else
    backtracking_flag = 0;
    assert(isa(f.L, 'numeric'), 'f.L must be numeric');
    assert(sum(size(f.L) ~= 1) == 0, 'f.L must be a scalar');
end

% N
if isempty(N)
    assert(nargin > 3 && ~isempty(x0), ...
        'The initial point must be provided if N is not provided.');
    N = length(x0);
else
    assert(isa(N, 'numeric'), 'N must be numeric');
    assert(sum(size(N) ~= 1) == 0, 'N must be a scalar');
    N = round(N);
end

% x0
if (nargin < 3) || isempty(x0); x0 = zeros(N, 1); end
assert(N == length(x0), 'The initial point must have length N');

% param
if (nargin < 4); param = []; end
if ~isfield(param, 'TOL'); param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 1000; end

%% Initialization
x = x0; % Signal to find
y = x0; % Extrapolation of x
t = 1; % Time step
k = 0; % Iteration number
if backtracking_flag % If we don't know the Lipschitz constant of f(x)
    eta = 2;
    L = 1;
else
    L = (f.L); % Lipschitz constant of f(x)
end

difference = Inf; % Difference between solutions in successive iterations
Evals = zeros(1, param.MAX_ITER);
Evals(1) = g.eval(x) + f.eval(x);

%% Iterative steps of FISTA
while (difference > param.TOL) && (k < param.MAX_ITER)
    
    k = k + 1;
    
    % Update x
    x_old = x;
    x = y - f.grad(y)./L;
    x = g.prox(x, 1./L);
        
    % Backtracking Line Search
    if backtracking_flag
        while f.eval(x) > f.eval(y) + f.grad(y)' * (x - y) + ...
                (L./2) * norm(x - y, 2).^2
            L = eta .* L;
            x = y - f.grad(y)./L;
            x = g.prox(x, 1./L);
        end
    end
    
    % Update time step
    t_old = t;
    t = ( 1 + sqrt(1 + 4*t^2) ) / 2.0;
        
    % Update y
    y = x + ( (t_old - 1) / t ) * (x - x_old);
    
    % Compute the objective value at the new point
    Evals(k + 1) = g.eval(x) + f.eval(x);
    
    % Update difference
    difference = norm(x - x_old, 2);
end

% Trim down the values vector:
Evals = Evals(1:k+1);

end

