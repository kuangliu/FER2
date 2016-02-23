function varargout = bn_layer(layer, varargin)
%BN_LAYER Batch normalization layer with parameters gamma&beta
% implementation of papar:
%       http://arxiv.org/pdf/1502.03167.pdf
% 
% Input X is sized [D, N], each column is a sample.
%
% Forward pass: it performs Unit Gaussian Normalization:
%       1. X_norm = (X-mean(X))/std(X);
%       2. y = gamma * X + beta
%   where gamma & beta are [D,1] vectors
%
%   - In training mode: [y, layer] = bn_layer(layer) 
%       we keep tracking the running mean/std for predict use.
%   - In test mode: y = bn_layer(layer)
%       we use the tracked mean/std to compute output
%   
%
% Backward pass: [dX, dGamma, dBeta] = bn_layer(layer, dy)
%   it computes local gradient. See paper for more math details.
%


if nargin == 1 || isempty(varargin)
    % forward pass
    X = layer.X;
    
    % initialize params for the first time
    if ~isfield(layer, 'gamma')
        D = size(X,1);
        layer.gamma = rand(D, 1); % init scale param
        layer.beta = zeros(D, 1); % shift param
        layer.running_mean = zeros(D, 1);
        layer.running_std = ones(D, 1);
    end
    
    gamma = layer.gamma;
    beta = layer.beta;
    mode = layer.mode; % train/test
    
    if strcmp(mode, 'train') == 1 
        % training mode, keep tracking the running mean/std
        momentum = 0.1;
        
        X_mean = mean(X, 2);
        layer.running_mean = (1-momentum)*layer.running_mean + momentum*X_mean;
        
        X_std = std(X, 1, 2);
        layer.running_std = (1-momentum)*layer.running_std + momentum*X_std; 
        
        % unit Guassian normalization: X_norm = (X-mean(X))/std(X)
        X_center = bsxfun(@minus, X, X_mean);
        X_norm = bsxfun(@rdivide, X_center, X_std+eps); % add eps for preventing divide by 0 error
        
        % save X_norm for BP usage
        layer.X_norm = X_norm;
        
        % compute activations y = gamma*X_norm+beta
        varargout{1} = bsxfun(@plus, bsxfun(@times, X_norm, gamma), beta);
        varargout{2} = layer;
    else
        % test mode, use running mean/std to compute output
        X_mean = layer.running_mean;
        X_std = layer.running_std;
        X_center = bsxfun(@minus, X, X_mean);
        X_norm = bsxfun(@rdivide, X_center, X_std+eps);
        
        varargout{1} = bsxfun(@plus, bsxfun(@times, X_norm, gamma), beta);
    end
else
    % backward pass
    X = layer.X;
    X_norm = layer.X_norm;
    gamma = layer.gamma;
    dy = varargin{1};
    
    VarX = var(X, 1, 2);
    X_center = bsxfun(@minus, X, mean(X, 2));
    
    dX_norm = bsxfun(@times, dy, gamma);
    
    dVarX = -0.5*dX_norm.*bsxfun(@rdivide, X_center, (VarX+eps).^1.5);
    dVarX = sum(dVarX, 2);
    
    dEX_tmp = bsxfun(@rdivide, dX_norm, (VarX+eps).^0.5);
    dEX = -sum(dEX_tmp, 2);
    
    N = size(X, 2);
    t = dEX_tmp + 2/N*bsxfun(@times, X_center, dVarX);
    dX = bsxfun(@plus, t, dEX/N);

    % compute dGamma & dBeta
    dGamma = sum(dy.*X_norm, 2);
    dBeta = sum(dy, 2);
    
    varargout{1} = dX;
    varargout{2} = dGamma;
    varargout{3} = dBeta; 
end


