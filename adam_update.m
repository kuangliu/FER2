function [W, state] = adam_update(W, dW, lr, state)
%ADAM_UPDATE
% implementation of paper: http://arxiv.org/pdf/1412.6980.pdf
%
% Inputs:
%   - W: weights to be updated
%   - dW: the weight gradients
%   - state: a struct containing the state of the optimizer; after each
%            call the state is modified
%
% Outputs:
%   - W: weights after update
%   - state: state after update
%


if ~isfield(state, 'm')  % init params for the first time
    state.m = zeros(size(W), 'single');    
    state.v = zeros(size(W), 'single');    
    state.t = 0;
end

beta1 = 0.9;
beta2 = 0.999;

state.m = beta1 * state.m + (1-beta1) * dW;
state.v = beta2 * state.v + (1-beta2) * (dW.*dW);
state.t = state.t + 1;

mb = state.m ./ (1 - beta1^state.t); % bias correation
vb = state.v ./ (1 - beta2^state.t);

W = W - lr*mb./(sqrt(vb) + 1e-8);



