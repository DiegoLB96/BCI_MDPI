function [varargout] = rbmDiscriminativeDeep(data, layerProp, classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   23.07.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBM Discriminative Trainig
%
% SYNTAX
% [layerProp] = rbmDiscriminative(data, layerProp, classes)
% does Discriminative Training
% 
% DEFINITION
% Does Discriminative Training with the resulting in the updates of the weights
% and biases for the Hidden and Classification layer (This function is layer 
% that do not have a classification layer and are used more as autoencoders)
%
% Abbreviations:
%%% INPUTS %%%
% data: A batch of data to train the RBM
% layerProp: Structure with the definitions of the RBM layer
% ngibbs: number of gibbs samples to use, standard for CD is 1
% classes: matrix with the classes for the batch data
%
%%% Outputs %%% 
% layerProp: it takes out the layerProp updated
%
% Note: This function may need to be adjusted in the future for different 
% proportion of the updates in the functions



% [nTrials nInputs] = size(data);
% dw = zeros(size(layerProp.weights));
% dcw = zeros(size(layerProp.classWeights));
% dhb =  sum(sigma(:,idx)) - sum(sigma.*repmat(probs(1:end),layerProp.numNeurons,1)); 



%%%% PREINITIALIZE WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%
numLayers = size(layerProp,2);
L = zeros(numLayers + 1 , 1 );
for i = 1 : numLayers
	L(i) = size(layerProp(i).weights,1);
    reactFunct(i) = layerProp(i).hidFunct;  
end
if ~strcmp(layerProp(1,i).trainingType,'RBMDeep')
    L(i+1) = layerProp(i).numNeurons;
else % Deep auto encoder % 
    for i = numLayers : -1 :1
        L(i) = size(layerProp(i).weights,2);
        reactFunct(i) = layerProp(i).visFunct;
    end
end
%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%

        
%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter=3; % Number of iterations %
if epoch < 6  % First update top-level weights holding other weights fixed.
    N = size(data,1);
    wtemp = [data];% ones(N,1)];
    for i = 1 : numLayers-1 % For all the layers except the output layer
        wtemp = [wtemp ones(N,1)]; % add ones for the biases 
        w = [layerProp(i).weights; layerProp(i).hidbiases];
        wtemp = layerProp(i).hidFunct.React(wtemp * w,1);
    end         
    % output layer
    w = [layerProp(numLayers).weights; layerProp(numLayers).hidbiases];
    VV = (w(:)')';
    Dim = [L(end-1); L(end)];
    [X, fX] = minimize(VV,'cgClassify',max_iter, reactFunct(end), Dim, wtemp,targets);
    w = reshape(X,L(end-1)+1,L(end));
    
    layerProp(numLayers).weights = w(1:L(numLayers),:);
    layerProp(numLayers).hidbiases = w(L(numLayers)+1:end, :);
else
    VV = [];
    for i = 1 : numLayers
        w = [layerProp(i).weights; layerProp(i).hidbiases];
        VV = [VV' w(:)']';
    end
    Dim = L;
    wtemp = [data];% ones(N,1)];
    [X, fX] = minimize(VV,'cgClassify',max_iter, reactFunct, Dim, wtemp,targets);
    
    xxx = 0;
    for i = 1 : numLayers
        w = reshape( X( xxx + 1 : xxx + (L(i)+1) * L(i+1) ), L(i)+1,L(i+1) );
        xxx = xxx + (L(i)+1) * L(i+1);
        
        layerProp(i).weights = w(1:L(i),:);
        layerProp(i).hidbiases = w(L(i)+1:end, :);
    end
    
end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varargout = {layerProp};
varargout = varargout(1:nargout);
end

