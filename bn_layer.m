function varargout = bn_layer(X, varargin)
%BN_LAYER batch normalization layer
% 
% Forward pass: y = bn_layer(X, gamma, beta)
%   it performs Unit Gaussian Normalization:
%       1. X = (X-mean(X))/std(X);
%       2. y = gamma * X + beta
%   where gamma&beta is [1,N] vectors
%
% Backward pass: [dX, dGamma, dBeta] = bn_layer(X, gamma, dy)
%   it compute gradients of parameters gamma&beta.
%       1. dX = dy*gamma*()
%       2. dGamma = dy*X
%       3. dBeta = dy
%
% Input X is sized [D, N], each column is a sample.

eps = 1e-5;
if nargin == 1 || isempty(varargin)
    % forward pass
    gamma = varargin{1};
    beta = varargin{2};
    % unit gaussian normalization X_norm = (X-mean(X))/std(X)
    X_norm = bsxfun(@rdivide,bsxfun(@minus,X,mean(X)),std(X)+eps);
    % compute activations y = gamma*X+beta
    varargout{1} = bsxfun(@plus,bsxfun(@times,X_norm,gamma),beta);
    
else
    % backward pass
    gamma = varargin{1};
    dy = varargin{2};
    dX = 
    
    
end


