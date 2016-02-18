function varargout = bn_layer(X, varargin)
%BN_LAYER batch normalization layer with parameters gamma&beta
% See the paper for more math details: 
% <Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift>
%
% Forward pass: [y, X_norm] = bn_layer(X, gamma, beta)
%   it performs Unit Gaussian Normalization:
%       1. X_norm = (X-mean(X))/std(X);
%       2. y = gamma * X + beta
%   where gamma & beta are [D,1] vectors
%
% Backward pass: [dX, dGamma, dBeta] = bn_layer(X, X_norm, gamma, dy)
%   it computes local gradient. See paper for more math details.
%   
% Input X is sized [D, N], each column is a sample.

eps = 1e-5;
if nargin == 3
    % forward pass
    gamma = varargin{1}; % [D,1]
    beta = varargin{2};  % [D,1]
    % unit gaussian normalization X_norm = (X-mean(X))/std(X)
    X_center = bsxfun(@minus, X, mean(X,2));
    X_norm = bsxfun(@rdivide, X_center, std(X,1,2)+eps);
    % compute activations y = gamma*X_norm+beta
    varargout{1} = bsxfun(@plus,bsxfun(@times,X_norm,gamma),beta);
    varargout{2} = X_norm;
    
else
    % backward pass
    X_norm = varargin{1};
    gamma = varargin{2};
    dy = varargin{3};
    
    VarX = var(X,1,2);
    X_center = bsxfun(@minus, X, mean(X,2));

    dX_norm = bsxfun(@times, dy, gamma);   % TODO: dXnorm = dy * r;
    
    dVarX = -0.5 * dX_norm .* bsxfun(@rdivide, X_center, (VarX+eps).^1.5);
    dVarX = sum(dVarX,2);
    
    dEX_tmp = bsxfun(@rdivide, dX_norm, (VarX+eps).^0.5);
    dEX = -sum(dEX_tmp,2);
    
    N = size(X,2);
    t = dEX_tmp + 2/N*bsxfun(@times, X_center, dVarX);
    dX = bsxfun(@plus, t, dEX/N);
    
    % compute dGamma & dBeta
    dGamma = sum(dy .* X_norm, 2);
    dBeta = sum(dy, 2);
    
    varargout{1} = dX;
    varargout{2} = dGamma;
    varargout{3} = dBeta;
    
end


