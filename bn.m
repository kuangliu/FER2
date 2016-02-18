function y = bn(X, varargin)
%BN Batch Normalization without parameters gamma&beta
%
% Forward: y = bn(X) it computes
%       y = (X-mean(X))/std(X)
%
% Backward: dX = bn(X, dy)
%       compute input gradient dX given output gradient dy
%       

    
if nargin == 1 || isempty(varargin)
    % forward pass: y = (X-mean(X))/std(X)
    X_center = bsxfun(@minus, X, mean(X,2));
    y = bsxfun(@rdivide, X_center, std(X,1,2));

else
    % backward pass: compute dX given dy
    VarX = var(X,1,2);
    X_center = bsxfun(@minus, X, mean(X,2));
    
    dy = varargin{1};
    dX_norm = dy;   % TODO: dXnorm = dy * r;
    
    dVarX = -0.5 * dX_norm .* bsxfun(@rdivide, X_center, VarX.^1.5);
    dVarX = sum(dVarX,2);
    
    dEX_tmp = bsxfun(@rdivide, dX_norm, VarX.^0.5);
    dEX = -sum(dEX_tmp,2);
    
    N = size(X,2);
    t = dEX_tmp + 2/N*bsxfun(@times, X_center, dVarX);
    y = bsxfun(@plus, t, dEX/N); % y = dX
    
end

