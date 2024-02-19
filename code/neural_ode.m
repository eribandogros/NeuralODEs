%exact dynamics
z0 = [2; 0]; %initial value
%z0 = [0; 2];
A = [-0.1 -1; 1 -0.1]; 
%A = [-0.1, 2; -2, -0.1];
f = @(t,z) A*z;

numtimesteps = 1000;
t0 = 0;
t1 = 20;
t = linspace(t0, t1, numtimesteps);
[t, ztrain] = ode45(f, t, z0);
ztrain = ztrain';

% figure
% plot(ztrain(1,:),ztrain(2,:))
% title("Exact Dynamics") 
% xlabel("z(1)") 
% ylabel("z(2)")
% grid on

%model params
numodesteps = 50;
dt = t(2); %change in t
timesteps = (0:numodesteps)*dt;

neuralOdeParameters = struct;

stateSize = 2; %input/output dim
hiddenSize = 20; %can experiment with

neuralOdeParameters.fc1 = struct;
neuralOdeParameters.fc1.Weights = initialize(hiddenSize, stateSize);
neuralOdeParameters.fc1.Bias = initializeZeros([hiddenSize 1]);

neuralOdeParameters.fc2 = struct;
neuralOdeParameters.fc2.Weights = initialize(stateSize, hiddenSize);
neuralOdeParameters.fc2.Bias = initializeZeros([stateSize 1]);

%Display the learnable parameters of the model.
%neuralOdeParameters.fc1
%neuralOdeParameters.fc2

%Training opts
gradDecay = 0.9; %for Adam
sqGradDecay = 0.999;
learnRate = 0.002;

numIter = 150; %train iter, epochs
miniBatchSize = 20;

plotFrequency = 10; %phase diagram plotting
f = figure;
f.Position(3) = 2*f.Position(3);

subplot(1,2,1)
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

averageGrad = []; %for Adam
averageSqGrad = [];

numTrainingTimesteps = numtimesteps;
trainingTimesteps = 1:numTrainingTimesteps;
plottingTimesteps = 2:numtimesteps;

start = tic;

%training loop
for iter = 1:numIter
    
    % batching 
    [X, targets] = createMiniBatch(numTrainingTimesteps, numodesteps, miniBatchSize, ztrain);
    
    % loss and gradients
    % dlgradient called here and need to write this function from scratch!
    [loss,gradients] = dlfeval(@modelLoss,timesteps,X,neuralOdeParameters,targets);

    % update model 
    [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    
    %plot loss
    subplot(1,2,1)
    
    currentLoss = double(loss);
    addpoints(lineLossTrain,iter,currentLoss);
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Elapsed: " + string(D))
    drawnow
  
    % plot dynamics
    if mod(iter,plotFrequency)==0 || iter == 1
        subplot(1,2,2)

        % ode solver
        y = dlode45(@odeModel,t,dlarray(z0),neuralOdeParameters,DataFormat="CB");
        
        plot(ztrain(1,plottingTimesteps),ztrain(2,plottingTimesteps),"r--")
        hold on
        plot(y(1,:),y(2,:),"b-")
        hold off
        xlabel("x(1)")
        ylabel("x(2)")
        title("Predicted and Exact Dynamics")
        legend("Exact", "Predicted")
        drawnow
        %saveas(f,"dyn"+iter)        
   end
end

