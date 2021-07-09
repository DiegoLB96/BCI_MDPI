function [varargout] = rbmDiscriminative(data, layerProp, classes, trainPorcentage)
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
% and biases for the Hidden and Classification layer (This function is only 
% available for the last layer of training when a class exist)
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

targets = classes;

%%%% PREINITIALIZE WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%
numLayers = size(layerProp,2);
L = zeros(numLayers + 2 , 1 );
for i = 1 : numLayers
	L(i) = size(layerProp(i).weights,1);
    reactFunct(i) = layerProp(i).hidFunct;  
end
L(i+1) = size(layerProp(i).classWeights,1);
reactFunct(i+1) = layerProp(i).classFunct;
L(i+2) = layerProp(i).numClasses;
%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%

        
%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter=3; % Number of iterations %
if trainPorcentage < .1  % First update top-level weights holding other weights fixed(10% of training epochs).
    N = size(data,1);
    wtemp = [data];% ones(N,1)];
    for i = 1 : numLayers 
        wtemp = [wtemp ones(N,1)]; % add ones for the biases 
        w = [layerProp(i).weights; layerProp(i).hidbiases];
        wtemp = layerProp(i).hidFunct.React(wtemp * w,1);
    end         
    % output layer
    w = [layerProp(numLayers).classWeights; layerProp(numLayers).classBiases];
    VV = (w(:)')';
    Dim = [L(end-1); L(end)];
    [X, fX] = minimize(VV,'cgClassify',max_iter, reactFunct(end), Dim, wtemp,targets);
    w = reshape(X,L(end-1)+1,L(end));
    
    layerProp(numLayers).classWeights = w(1:L(numLayers+1),:);
    layerProp(numLayers).classBiases = w(L(numLayers+1)+1:end, :);
else
    VV = [];
    for i = 1 : numLayers
        w = [layerProp(i).weights; layerProp(i).hidbiases];
        VV = [VV' w(:)']';
    end
    w = [layerProp(i).classWeights; layerProp(i).classBiases];
    VV = [VV' w(:)']';
    
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
    w = reshape( X( xxx + 1 : xxx + (L(i+1)+1) * L(i+2) ), L(i+1)+1,L(i+2) );
    xxx = xxx + (L(i+1)+1) * L(i+2);
    layerProp(numLayers).classWeights = w(1:L(i+1),:);
    layerProp(numLayers).classBiases = w(L(i+1)+1:end, :);
end
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varargout = {layerProp};
varargout = varargout(1:nargout);
end

