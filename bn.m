function varargout = bn(X, varargin)
%BN 此处显示有关此函数的摘要
%   此处显示详细说明
    
if nargin == 1 || isempty(varargin)
    % forward, output X_norm = (X-mean(X))/std(X)
    X_center = bsxfun(@minus, X, mean(X,2));
    X_norm = bsxfun(@rdivide, X_center, std(X,1,2));
    varargout{1} = X_norm;

else
    N = size(X,2);
    VarX = var(X,1,2);
    X_center = bsxfun(@minus, X, mean(X,2));
    
    dy = varargin{1};
    dX_norm = dy;   % TODO: dXnorm = dy * r;
    
    dVarX = -0.5 * dX_norm .* bsxfun(@rdivide, X_center, VarX.^1.5);
    dVarX = sum(dVarX,2);
    
    dEX_tmp = bsxfun(@rdivide, dX_norm, VarX.^0.5);
    dEX = -sum(dEX_tmp,2);
    
    t = dEX_tmp + 2/N*bsxfun(@times, X_center, dVarX);
    dX = bsxfun(@plus, t, dEX/N);
    
    varargout{1} = dX;

end

