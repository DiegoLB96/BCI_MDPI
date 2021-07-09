function [varargout] = somInitLayers(nInputs, neuronPerLayer, trainEpochs, ...
    LR, Sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INIT LAYERS
% Initilize a structure for the SOM
%
% SYNTAX
% [layerProp] = somInitLayers(nInputs, trainingType, neuronPerLayer, typeVisible, typeLayer, ...
%     LR, CT, MOM, ...
%     typeClass, nClasses, LRc, CTc, MOMc,...
%     totStd)
% 
% DEFINITION
% Creates a structure with N-number of hidden layers for the RBM, structure
% needed for RBM train
% 
% %% Inputs %%
% nInputs: Specify the number of neurons needed as Input vector(Real value)%           
% neuronPerLayer: row vector with the number of neurons per Hidden layer(Real value)
%           If not specified it takes 50 neurons. If empty it takes one
%           order in magnitude(base 2) bigger than nInputs
% typeVisible: string with the form of the first layer(Visible layer)
%             [Binary,Rlin, Gauss, Softmax, noState]if empty or or not specified
%             it fills the layers with Binary form
% typeLayer: cell of strings with the form of the Hidden layers  
%             [Binary,Rlin, Gauss, Softmax, noState], if empty or or not specified
%             it fills the layers with Binary form
% trainEpochs: the number of training epochs per layer
% LR: specify the learning rate. Has to be inserted in a cell as numeric or as a function 
%               See rbmConvertRates 
% CT: specify the costs. Has to be inserted in a cell as numeric or as a function 
%               See rbmConvertRates 
%
% %% Outputs %%
% layerProp: structure with the initilized layers parameters


if nargin < 1, error('Needs at least the training type and number of features'); end
if nargin < 2, 
    neuronPerLayer = 50; 
elseif isempty(neuronPerLayer),  
    nNeurons = nextpow2(nInputs); 
    neuronPerLayer = [2^(nNeurons)]; 
    if neuronPerLayer*.7 < nInputs, neuronPerLayer = neuronPerLayer * 2; end
end
if nargin < 3, trainEpochs = []; end
if isempty(trainEpochs), trainEpochs = 100; end
if nargin < 4, LR = 0.1; end % learning rates
if nargin < 5, Sigma = floor(sqrt(neuronPerLayer)/2); end % costs

layerProp = struct('weights',[],...
    'trainEpochs',0,...
    'eta',0,'etaActual',0, 'tauEta',0,...% learning rate and constant for updating LR
    'sigma',0,'sigmaActual',0, 'tauSigma',0); % distance from BMU

layerProp.weights = rand(neuronPerLayer,nInputs);
layerProp.trainEpochs = trainEpochs;

layerProp.eta = LR;
layerProp.etaActual = layerProp.eta;
layerProp.tauEta = 100;% trainEpochs/log(LR);
layerProp.sigma = Sigma;
layerProp.sigmaActual = layerProp.sigma;
layerProp.tauSigma = trainEpochs/log(Sigma);

varargout = {layerProp};
varargout = varargout(1:nargout);
end