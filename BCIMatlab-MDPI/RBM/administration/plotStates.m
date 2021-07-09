function numFig = plotStates(layerProp, trainData, numFig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot the states of the weights and biases of the layers, and plot the 
% states of the class layer if available
% 

if nargin < 2 || numFig < 0, numFig = 1; end

numLayers = size(layerProp,2);

for i = 1 : numLayers
    figure(numFig + i) % Weights
        subplot(3,1,1)
        set(gca,'xtick',[],'ytick',[])
        hist(layerProp(1,i).weights)
        subplot(3,1,2)
        set(gca,'xtick',[],'ytick',[])
        hist(layerProp(1,i).hidbiases)
        subplot(3,1,3)
        set(gca,'xtick',[],'ytick',[])
        hist(layerProp(1,i).visbiases)
end

numFig = numFig + numLayers + 1;

if ~isempty(layerProp(1,end).typeClass)
    
    figure(numFig)
        subplot(2,1,1)
        set(gca,'xtick',[],'ytick',[])
        hist(layerProp(1,i).classWeights)
        subplot(2,1,2)
        set(gca,'xtick',[],'ytick',[])
        hist(layerProp(1,i).classBiases)
end

numFig = numFig + 1;
data = trainData(:,:,1);
for i = 1 : numLayers
    weights = layerProp(1,i).weights; hidbiases = layerProp(1,i).hidbiases; hidFunct = layerProp(1,i).hidFunct;
    figure(numFig)
        subplot(numLayers,1,i)
        IHidden = hidFunct.Input(weights,hidbiases,data,0,0); % Input for hidden units %
        RHidden = hidFunct.React(IHidden);  % Reaction of hidden units %
        SHidden = hidFunct.State(RHidden,IHidden); % States of hidden units %
        image(SHidden*256)
        set(gca,'xtick',[],'ytick',[])
        data = RHidden;
end


numFig = numFig + 2;
end