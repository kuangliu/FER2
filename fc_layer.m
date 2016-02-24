function varargout = fc_layer(layer, varargin)
%FC_LAYER fully connected layer
%
% It performs:
%   - forward pass: y = fc_layer(layer)
%   - backward pass: [dX, dW, db] = fc_layer(layer, dy)
% 
% Inputs:
%   - layer: FC layer with
%       - X: input data [D, N]
%       - W: weights [C, D]
%       - b: bias [C, 1]
%   - varargin: when perform backward pass, it is output gradient dy
%
% Outputs:
%   - varargout
%       - forward pass: return activations y = W*X+b
%       - backward pass: return local gradients dX, dW and db


if nargin == 1 || isempty(varargin)
    % forward pass, compute activations y
    varargout{1} = bsxfun(@plus, layer.W * layer.X, layer.b);
else
    % backward pass, compute local gradients dW & dX
    dy = varargin{1};
    varargout{1} = layer.W' * dy;     % dX
    varargout{2} = dy * layer.X';     % dW
    varargout{3} = sum(dy, 2);     % db
end




