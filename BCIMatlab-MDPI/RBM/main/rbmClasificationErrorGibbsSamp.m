function [varargout] = rbmClasificationErrorGibbsSamp(batchData, batchTargets, layerProp, nGibbs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.03.2010 - 31.05.2011 last modified 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Error calculation with different gibbs samplin, it uses intrincic saved
% function to calculate the values of input and of reactions
% it uses the same Gibbs sampling as the one used to train the last layer
% of the RBM 
% 

if nargin < 4, nGibbs = layerProp(end).ngibbs; end
if nargin < 3, error('Not possible to classiffy the error'); end

err_cr = 0;
counter = 0;
classCount = 0;

[nTrials, ~, nbt] = size(batchData); % [nTrials numdims nbt] = size(batchdata);

numClasses = size(batchTargets,2);

for batch = 1:nbt
    data = (batchData(:,:,batch));
    target = (batchTargets(:,:,batch));
    targetout = rbmGenerateLabel(data, layerProp, nGibbs);
    
    [C, J] = max(targetout,[],2); if C == 0, J = 0; end 
    [C, J1] = max(target,[],2); if C == 0, J1 = 0; end 
%     if(mod(batch,4) == 0), figure, hist(J); hold on, hist(J1); end
    counter = counter + length( find( J == J1 ) );
    classCount = classCount + histc(J, 1:numClasses); % number of class apperance
    err_cr = err_cr + sum(sum( (target - targetout).^2 )); % classification distance
end
err = (nTrials*nbt-counter)/(nTrials*nbt); % Number of misclasification data
classCount = classCount/(nTrials*nbt); % percentage per number of classes detected
err_cr = err_cr/(nTrials*nbt*numClasses);


% J and J1 are the generated class and the defined class
varargout = {err, err_cr, classCount};
varargout = varargout(1:nargout);

end