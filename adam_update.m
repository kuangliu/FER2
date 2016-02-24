function [X, state] = adam_update(X, dX, lr, state)
%ADAM_UPDATE
% implementation of paper: http://arxiv.org/pdf/1412.6980.pdf
%
% Inputs:
%   - X: variable to be updated
%   - dX: the gradients
%   - state: a struct containing the state of the optimizer; after each
%            call the state is modified
%
% Outputs:
%   - X: variable after update
%   - state: state after update
%


if ~isfield(state, 'm') % init params for the first time
    state.m = zeros(size(X), 'single');    
    state.v = zeros(size(X), 'single');    
    state.t = 0;
end

beta1 = 0.9;
beta2 = 0.999;

state.m = beta1 * state.m + (1-beta1) * dX;
state.v = beta2 * state.v + (1-beta2) * (dX.*dX);
state.t = state.t + 1;

mb = state.m ./ (1 - beta1^state.t); % bias correation
vb = state.v ./ (1 - beta2^state.t);

X = X - lr*mb./(sqrt(vb) + 1e-8);



