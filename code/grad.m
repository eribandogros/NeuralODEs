function g = grad(~,X0,neuralOdeParameters)

dfdw1 = neuralOdeParameters.fc2.Weights'*neuralOdeParameters.fc1.Weights*X0;
dfdb1 = neuralOdeParameters.fc2.Weights;
dfdw2 = neuralOdeParameters.fc1.Weights*X0+neuralOdeParameters.fc1.Bias;

g = -(1/2)*dfdw1;

end