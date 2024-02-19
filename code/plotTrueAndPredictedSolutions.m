function plotTrueAndPredictedSolutions(xTrue,xPred)

xPred = squeeze(xPred)';

err = mean(abs(xTrue(2:end,:) - xPred), "all");

plot(xTrue(:,1),xTrue(:,2),"r--",xPred(:,1),xPred(:,2),"b-",LineWidth=1)

title("Absolute Error = " + num2str(err,"%.4f"))
xlabel("x(1)")
ylabel("x(2)")

xlim([-2 3])
ylim([-2 3])

legend("Ground Truth","Predicted")

end