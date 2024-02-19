%normalized Xavier weight initialization
function weights = initialize(out,in)

r = rand([out in]);
u = sqrt(6 / (in + out));
weights = -u+r*(2*u);
weights = dlarray(weights);

end