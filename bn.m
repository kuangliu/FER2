function varargout = bn(X, varargin)
%BN 此处显示有关此函数的摘要
%   此处显示详细说明
    
if nargin == 1 || isempty(varargin)
    % forward, output X_norm = (X-mean(X))/std(X)
    X_center = bsxfun(@minus, X, mean(X));
    X_norm = bsxfun(@rdivide, X_center, std(X));
    varargout{1} = X_norm;

else
    D = size(X,1);
    
    dEX = ones(size(X))/D;  % dEX is dEX/dX, same size as X
    X_center = bsxfun(@minus, X, mean(X));
    
    VarX = var(X);
    dVarX = 2/D * X_center.*(1-dEX); % dVarX is dVarx/dX, same size as X
    
    dy = varargin{1};
    t = -0.5*bsxfun(@rdivide, X_center.*dVarX, VarX.^1.5);
    dX = dy .* (t + bsxfun(@rdivide, 1-dEX, VarX.^0.5));
    varargout{1} = dX;

end

