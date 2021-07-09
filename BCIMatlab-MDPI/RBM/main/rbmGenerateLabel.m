function varargout = rbmGenerateLabel(data, layerProp, nGibbs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 31.05.2011 - last modified 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates the label from a Data vector, it uses the the gibbs sampling
% if its not defined it takes the used to train the last layer of the RBM
% 
if nargin < 3, nGibbs = layerProp(end).ngibbs; end

numLayers = size(layerProp,2);
targetOut = data;
for j = 1 : numLayers
    %%%%%%%%% POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    IHidden = layerProp(j).hidFunct.Input(layerProp(j).weights,layerProp(j).hidbiases,targetOut,0,0);
    RHidden = layerProp(j).hidFunct.React(IHidden);  % Reaction of hidden units %
    if j~=numLayers, targetOut = RHidden;end
end

%%%%%%%%% Markov Chain (Gibbs sampling)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
targetOutTemp = 0;
for ng = 1 : nGibbs
    IClass = layerProp(end).classFunct.Input(layerProp(end).classWeights,layerProp(end).classBiases,RHidden);
    RClass = layerProp(end).classFunct.React(IClass);
    %         SClass = layerProp(end).classFunct.State(RClass,IClass);
    if ng ~= layerProp(end).ngibbs % Just to save some time
        IHidden = layerProp(end).hidFunct.Input(layerProp(end).weights,layerProp(end).hidbiases,targetOut,RClass,layerProp(end).classWeights);
        RHidden = layerProp(end).hidFunct.React(IHidden);  % Reaction of hidden units %
    end
    targetOutTemp = targetOutTemp + RClass;
end
targetOut = targetOutTemp/layerProp(end).ngibbs;

varargout = {targetOut};
varargout = varargout(1:nargout);
end
