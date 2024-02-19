function g = grad(~,X0,neuralOdeParameters)

%need to return -a(t)^T dfd\theta; dfd\theta means dfdw1, dfdb1, dfdw2, dfdb2
%where f(X) = w2*(w1*X+b1)+b2
%a(t) = dLdX where L = 1/(2*batchsize*odetimesteps)*abs(X-targets)
%trying a simple example where L=0.25*abs(X-targets)
%a = 0.25;
%switched to L2 norm
%dLdX = 2*abs(X0-targets)
dfdw1 = neuralOdeParameters.fc2.Weights'*neuralOdeParameters.fc1.Weights*X0;
dfdb1 = neuralOdeParameters.fc2.Weights;
dfdw2 = neuralOdeParameters.fc1.Weights*X0+neuralOdeParameters.fc1.Bias;
%dfdb2 = 1; % will cause error w odesolve
g = -(1/2)*dfdw1;

end