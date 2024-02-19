function [loss,gradients] = modelLoss(tspan,X0,neuralOdeParameters,targets)

% predictions, forwar
% pass model through ode solver
% function to solve must be function handle, @odeModel
X = dlode45(@odeModel,tspan,X0,neuralOdeParameters,DataFormat="CB",GradientMode="adjoint");%data format of X0, channel then batch

% L1 loss
numele = X.size(1)*X.size(2)*X.size(3);
loss = sum(sum(sum(abs(X-targets))))/numele;
%loss = l2loss(X,targets,NormalizationFactor="all-elements",DataFormat="CBT");

% gradients
gradients = dlgradient(loss,neuralOdeParameters);

%adjoint ode solve
%need backwards through time, reverse tspan
tspan_new = [0.02 0]; %simple example, 1 time step
%dLdtheta = dlode45(@grad,tspan_new,X,neuralOdeParameters,DataFormat="CB");
X0
X
gradw = dlode45(@grad,tspan_new,X0,neuralOdeParameters,DataFormat="CB")
gradients.fc1.Weights
gradients.fc2.Weights
gradients.fc1.Bias
gradients.fc2.Bias

%Bowen's notes (when working with a tanh layer)
% inner = tanh(neuralOdeParameters.fc1.Weights*X0 + neuralOdeParameters.fc1.Bias);
% dfdZ =neuralOdeParameters.fc2.Weights*(1-tanh(inner)^2)*neuralOdeParameters.fc1.Weights; %?
% gradients.fc1.Weights = inner;
% [b_row,b_col] = size(neuralOdeParameters.fc2.Bias);
% gradients.fc2.Bias =eye(b_col); %we suppose all vectors all colum vectors
% gradients.fc1.Bias =neuralOdeParameters.fc2.Weights*(1-tanh(inner)^2);

%%%%In order to use ode solver to get gradients, I need to know what is dL/dX,
%%%%dL/dparameters, X(tspan2) at tspan(2), where dL/dX dL/dparameters at tspan(2)
%%%% should be given in arguments of function modelLoss
%%%% gradients = odesolver(X(tspan(2),aug_dynamics,tspan(2),tspan(1))
%%%%
X_
end