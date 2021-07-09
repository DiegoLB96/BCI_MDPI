function [varargout] = rbmClasificationError(batchData, batchTargets, layerProp)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normal error classification with 1 gibbs sampling, it uses adding one
% column of ones on the data to work with the biases uses the functions to
% calculate the states
% **** CURRENTLY NOT IN USE Replace with rbmClassificationErrorGibbsSamp *****

if nargin < 3, error('Not possible to classiffy the error'); end

err_cr = 0;
counter = 0;
classCount = 0;

[nTrials, ~, nbt] = size(batchData); % [nTrials numdims nbt] = size(batchdata);
numLayers = size(layerProp,2);
numClasses = size(batchTargets,2);

for batch = 1:nbt
    data = (batchData(:,:,batch));
    target = (batchTargets(:,:,batch));
    targetout = [data ones(nTrials,1)];
    %%%%%%%%% FEATURE EXTRACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : numLayers
        w = [layerProp(i).weights; layerProp(i).hidbiases];
        targetout = layerProp(i).hidFunct.React(targetout * w,1);
        targetout = [targetout ones(nTrials,1)];
    end
    w = [layerProp(i).classWeights; layerProp(i).classBiases];
    targetout = layerProp(i).classFunct.React(targetout * w,1);

    [~, J] = max(targetout,[],2); % [I J] = max(targetout,[],2);
    [~, J1] = max(target,[],2); % [I1 J1] = max(target,[],2);
%     if(mod(batch,4) == 0), figure, hist(J); hold on, hist(J1); end
    counter = counter + length( find( J == J1 ) );
    classCount = classCount + histc(J, 1:numClasses); % number of class apperance
    err_cr = err_cr - sum(sum( target(:,1:end).*log(targetout) ) ) ; % classification distance
end
err = (nTrials*nbt-counter)/(nTrials*nbt); % Number of misclasification data
classCount = classCount/(nTrials*nbt); % percentage per number of classes detected
err_cr = err_cr/nbt;

varargout = {err, err_cr, classCount};
varargout = varargout(1:nargout);

end