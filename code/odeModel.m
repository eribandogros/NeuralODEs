function f = odeModel(~,X0,neuralOdeParameters)

% input -> linear -> tanh -> linear -> output

%f = tanh(neuralOdeParameters.fc1.Weights*X0 + neuralOdeParameters.fc1.Bias); %can try other activations here (a diff ReLU?)
f = neuralOdeParameters.fc1.Weights*X0 + neuralOdeParameters.fc1.Bias;
f = neuralOdeParameters.fc2.Weights*f + neuralOdeParameters.fc2.Bias;

end