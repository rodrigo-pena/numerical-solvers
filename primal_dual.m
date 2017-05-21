function [x, energy] = primal_dual(F, G, K, N, x0, param)
% PRIMAL_DUAL solves the following nonlinear primal problem with
% Chambolle and Pock's primal-dual algorithm:
%
% (1)                  min_x (F(Kx) + G(x)) = min_x E(x),
%
% where both F(x) and G(x) are convex. The algorithm converges faster if
% G(x) is uniformly convex, and its Lipschitz constant is specified. It
% converges even faster if both F(x) and G(x) are uniformly convex, and
% their Lipschitz constants are specified.
%
%   Usage:
%       [x, energy] = primal_dual(F, G, K, N, x0, param)
%
%   Input:
%       F       : A Matlab structure containing information about the
%                 function F. It must have as fields:
%           F.eval  : A function handle with the expression for F(x).
%           F.prox  : A function handle with the proximity operator of
%                     sigma*F(x), as a function of x and sigma.
%                     (e.g.: if F.eval = @(x) norm(x, 1), then
%                      F.prox = @(x, sigma) shrinkage(x, sigma)).
%           F.L     : (Optional) Lipschitz constant of F, if F is uniformly
%                     convex. Otherwise, leave it empty.
%       G       : A Matlab structure containing information about the
%                 function G. It must have as fields:
%           G.eval  : A function handle with the expression for G(x).
%           G.prox  : A function handle with the proximity operator of
%                     tau*G(x), as a function of x and tau.
%                     (e.g.: if G.eval = @(x) norm(x, 1), then
%                      G.prox = @(x, tau) shrinkage(x, tau)).
%           G.L     : (Optional) Lipschitz constant of f, if f is uniformly
%                     convex. Otherwise, leave it empty.
%       K       : A matrix encoding a map K: X -> Y, where X and Y are two
%                 vector spaces, and x belongs to X.
%       N       : Length of the vector x to be learned.
%                 (Default: length(x0))
%       x0      : Initialization of the vector x to be learned.
%                 (Default: zeros(N,1))
%       param   : Matlab structure with some additional parameters.
%           param.TOL       : Stop criterium. When ||x(n) - x(n-1)||_2,
%                             where n is the iteration number, is less than
%                             TOL, we quit the iterative process.
%                             (Default: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the
%                             iterative process.
%                             (Default: 1000).
%           param.constraint: A function handle imposing some constraint on
%                             x.
%                             (Default: @(x) x).
%
%   Output:
%       x       : A N-by-1 vector with the solution to the optimization
%                 problem (1).
%       energy  : A vector with the energies E(x(n-1)), where n is the
%                 iteration number.
%
%   Example:
%       [x, energy] = primal_dual(F, G, K, N, x0, param)
%
%   See also: learn_sparse_signal.m, FISTA.m
%
%   References:
%       [1]	A. Chambolle and T. Pock, "A First-Order Primal-Dual Algorithm
%       for Convex Problems with Applications to Imaging," J Math Imaging
%       Vis, vol. 40, no. 1, pp. 120-145, Dec. 2010.
%
% Author: Rodrigo Pena
% Date: 8 Dec 2015
% Testing: demo_sparse_signal_learning.m

%% Parse input
% F
assert(isfield(F, 'eval') && isfield(F, 'prox'), ...
    'F doesn''t have the correct fields.');
assert(isa(F.eval, 'function_handle'), 'F.eval must be a function handle');
assert(isa(F.prox, 'function_handle'), 'F.prox must be a function handle');
acceleration = 0;
delta = 0;
if isfield(F, 'L')
    if ~isempty(F.L);
        assert(isa(F.L, 'numeric'), 'F.L must be numeric');
        assert(sum(size(F.L)~=1) == 0, 'F.L must be a scalar');
        acceleration = acceleration + 1;
        delta = 1./F.L;
    end
end

% G
assert(isfield(G, 'eval') && isfield(G, 'grad'), ...
    'G doesn''t have the correct fields.');
assert(isa(G.eval, 'function_handle'), ...
    'G.eval must be a function handle');
assert(isa(G.grad, 'function_handle'), ...
    'G.grad must be a function handle');
gamma = 0;
if isfield(G, 'L')
    if ~isempty(G.L);
        assert(isa(G.L, 'numeric'), 'G.L must be numeric');
        assert(sum(size(G.L)~=1) == 0, 'G.L must be a scalar');
        acceleration = acceleration + 1;
        gamma = 1./G.L;
    end
end

% N
if isempty(N)
    assert(nargin > 4, 'x0 must be provided if N is not provided.');
    assert(~isempty(x0), 'x0 must be provided if N is not provided.');
    N = length(x0);
else
    assert(isa(N, 'numeric'), 'N must be numeric');
    assert(sum(size(N)~=1) == 0, 'N must be a scalar');
    N = round(N);
end

% K
assert(size(K, 2) == N, 'size(K, 2) must be equal to N');

% x0
if (nargin < 5) || isempty(x0); x0 = zeros(N,1); end
assert(N == length(x0), 'x0 must have length N');

% param
if (nargin < 6); param = []; end
if ~isfield(param, 'TOL'); param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 1000; end
if ~isfield(param, 'constraint'); param.constraint = @(x) x; end

%% Initialization
x = x0; % Primal variable
x_bar = x0; % Auxiliary variable
y = K*x0; % Dual variable
switch acceleration % Initialize time steps accordingly
    case 0
        sigma = 1;
        tau = sigma;
        theta = mean([0, 1]);
    case 1
        sigma = 1./normest(K);
        tau = sigma;
        if (gamma == 0); gamma = delta; end
    case 2
        mu = 2*sqrt(gamma.*delta)./normest(K);
        sigma = mu./(2.*delta);
        tau = mu./(2.*gamma);
        theta = mean([1./(1 + mu), 1]);
end
n = 0; % Iteration number

difference = Inf; % Difference between solutions in successive iterations
energy = zeros(1, param.MAX_ITER);
energy(1) = F.eval(x) + G.eval(x);

%% Iterative steps
while (difference > param.TOL) && (n < param.MAX_ITER)
    
    n = n + 1;
    
    % Update dual variable y
    y = F.prox(y + sigma .* K * x_bar, sigma);
    
    % Update primal variable x
    x_old = x;
    x = G.prox(x - tau.* K' * y, tau);
    
    if acceleration == 1
        % Update time steps
        theta = 1 ./ sqrt(1 + 2 .* gamma .* tau);
        tau = theta .* tau;
        sigma = sigma ./ theta;
    end
    
    % Update auxiliary primal variable x_bar
    x_bar = x + theta.*(x - x_old);
    
    % Compute the energy
    energy(n + 1) = F.eval(x) + G.eval(x);
    
    % Update difference
    difference = norm(x - x_old, 2);
end

% Trim down energy vector:
energy = energy(1:n+1);

end

