% Find the position of the winner neuron
function [ind,dist] = somWinnerNeuron (xn,weights)
    dist = sum( sqrt((weights - repmat(xn,size(weights,1),1)).^2),2);
    %// find the winner
    [v ind] = min(dist);
end