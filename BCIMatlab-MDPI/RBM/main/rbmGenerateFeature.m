function varargout = rbmGenerateFeature(batchData, batchTargets, layerProp, nGibbs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 08.03.2010 - last modified 30.03.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruct the feature using Gibbs sampling on the bottom layer using
% a class label as input vector
% 

if nargin < 3, error('Not possible to reconstruct the signal'); end 
if nargin < 4, nGibbs = layerProp(end).ngibbs; end

[nTrials, nInputs, nbt] = size(batchData); % [nTrials numdims nbt] = size(batchdata);
numClasses = size(batchTargets,2);
numLayers = size(layerProp,2);

reconst =  zeros(numClasses, nInputs);
averageSignal =  zeros(numClasses, nInputs);
sizeClass = zeros(numClasses,1);

for batch = 1: nbt
    data = (batchData(:,:,batch));
    target = (batchTargets(:,:,batch));
    targetout = zeros(size(data,1),size(layerProp(end).weights,1));
    RHidden = 0;
    RClass = target;
    targetoutTemp = 0;
    %%%% Gibbs sampling %%%%
    for ng = 1 : nGibbs
        IHidden = layerProp(end).hidFunct.Input(layerProp(end).weights,layerProp(end).hidbiases,targetout,RClass,layerProp(end).classWeights);
        RHidden = layerProp(end).hidFunct.React(IHidden);
        IVisible = layerProp(end).visFunct.Input(layerProp(end).weights',layerProp(end).visbiases,RHidden);
        RVisible = layerProp(end).visFunct.React(IVisible); 
        targetoutTemp = targetoutTemp + RVisible;
        targetout = targetoutTemp;
    end
    targetout = targetoutTemp/ng;
    
    for j = numLayers-1:-1:1
        IVisible = layerProp(j).visFunct.Input(layerProp(j).weights',layerProp(j).visbiases,targetout);% Input for visible units % 
		RVisible = layerProp(j).visFunct.React(IVisible); 
        targetout = RVisible;
    end
    
    [~, J] = max(batchTargets,[],2);
    for j = 1 : numClasses
        idx = J == j;
        reconst(j,:) = reconst(j,:) + sum(targetout(idx,:),1);
        averageSignal(j,:) = averageSignal(j,:) + sum(data(idx,:),1);
        sizeClass(j) = sizeClass(j) + sum(idx);
    end
end

% Average %
for j = 1 : numClasses
    reconst(j,:) = reconst(j,:)./sizeClass(j);
    averageSignal(j,:) = averageSignal(j,:)./sizeClass(j);
end

varargout = {reconst, averageSignal};
varargout = varargout(1:nargout);

end