function [varargout] = rbmDiscriminative(data, layerProp, classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   03.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBM Discriminative Trainig
%
% SYNTAX
% [dvw, dhb, dvb, dcw, dcb, outData, err] = rbmConstrastiveD(data, layerProp, visFunct, ngibbs, classes, classLayerProp)
% does Constrastive Divergance
% 
% DEFINITION
% Does Discriminative Training with the resulting in the updates of the weights
% and biases for the Hidden and Classification layer
%
% Abbreviations:
%%% INPUTS %%%
% data: A batch of data to train the RBM
% layerProp: Structure with the definitions of the RBM layer
% ngibbs: number of gibbs samples to use, standard for CD is 1
% classes: matrix with the classes for the batch data
%
%%% Outputs %%% 
% dw: update for the weights of the RBM
% dhb: update for the hidden biases
% dvb: update for the visible biases
% dcw: update for the class weights 
% dcb: update for the class biases
% outData: data used for training consecutive RBMs
% err: error of the current CD optimization 

[nTrials nInputs] = size(data);

classesReal = realRepresentation(classes);
classWeights = layerProp.classWeights;
classBiases = layerProp.classBiases;
classFunct = layerProp.classFunct;

hidFunct  = layerProp.hidFunct;     %structure with the specified reaction,  
visFunct  = layerProp.visFunct;     %state and input functions for the different RBM
weights = layerProp.weights;        % weights of the current Hidden unit
hidbiases = layerProp.hidbiases;    % hidden biases for the current RBM
visbiases = layerProp.visbiases;    % visible biases for the current RBM

dw = zeros(size(layerProp.weights));
dcw = zeros(size(layerProp.classWeights));


% Compute gradient for each trial and accumulate into d_...
Oyj = repmat(layerProp.hidbiases, nTrials*layerProp.numClasses,1) +...
    reshape(repmat(data*layerProp.weights,layerProp.numClasses,1)',layerProp.numNeurons,nTrials*layerProp.numClasses)' + ...
    repmat(layerProp.classWeights,1,nTrials); % o_yj(x) = c + U + Wx
Oyj = hidFunct.Input(weights,hidbiases,data,classes,classWeights);
sigma = layerProp.hidFunct.React(Oyj);

%%%Compute the probabilities p(y|x) for multiple trials of x
r = repmat(exp(classBiases),nTrials,1).*repmat(prod(1.+exp(Oyj'))',1,layerProp.numClasses);
s = repmat(sum(r,2),1,layerProp.numClasses);
probs = (r./s);


idx = (1:layerProp.numClasses:nTrials*layerProp.numClasses)'+classesReal-1;

dhb =  sum(sigma(:,idx)) - sum(sigma.*repmat(probs(1:end),layerProp.numNeurons,1)); 

for k = 1:nTrials, 
    x = data(k,:);
    y = classes(k);
    
    mask = [1+layerProp.numClasses*(k-1) : layerProp.numClasses*(k)];
    sigmoid_O = sigma(mask,:);
    classprobs = probs(k,:);
    
    dcw(y,:) = dcw(y,:) + sigmoid_O(y,:);
    for j=1:layerProp.numClasses,
        dcw(j,:) = dcw(j,:) -classprobs(j)*sigmoid_O(j,:);
    end

    Q = repmat(classprobs*sigmoid_O,layerProp.featuredim,1);
    for j = 1:layerProp.featuredim, 
        Q(j,:) = x(j)*Q(j,:); 
    end
    
    dw = dw + x*sigmoid_O(y,:) - Q;    
end % for each trial


varargout = {dw, dhb, dcw};
varargout = varargout(1:nargout);
end

