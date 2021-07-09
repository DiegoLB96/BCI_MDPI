function [varargout] = BPInitLayers(nInputs, neuronPerLayer, numClasses, trainEpochs, ...
    LR, Mom)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INIT LAYERS
% Initilize a structure for the Back Propagation
%
% SYNTAX
% [layerProp] = BPInitLayers(nInputs, neuronPerLayer, numClasses,trainEpochs,
%     LR, MOM)
% 
% DEFINITION
% Creates a structure with N-number of hidden layers for the BP, structure
% needed for BP train
% 
% %% Inputs %%
% nInputs: Specify the number of neurons needed as Input vector(Real value)%           
% neuronPerLayer: row vector with the number of neurons per Hidden layer(Real value)
%           If not specified it takes 50 neurons. If empty it takes one
%           order in magnitude(base 2) bigger than nInputs
% numClasses: determine the number of classes to classify
% trainEpochs: the number of training epochs per layer
% LR: specify the learning rate. Has to be inserted in a cell as numeric or as a function 
% Mom: specify the Momentums. Has to be inserted in a cell as numeric or as a function 
%               
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
if nargin < 3, numClasses = 2; end
if nargin < 4, trainEpochs = []; end
if isempty(trainEpochs), trainEpochs = 100; end
if nargin < 5, LR = 0.1; end % learning rates
if nargin < 6, Mom = 0.1; end % costs

s = struct('weights',[],'weightsinc',[],'inputs',[],'outputs',[],...
    'trainEpochs',0,...
    'LR',0,'LRcurrent',0, ...
    'Mom',0,'Momcurrent',0); % distance from BMU

numLayers = size(neuronPerLayer,2) + 2; % input and output layer

layerProp = repmat(s, 1, numLayers);
neuronPerLayer(end+1) = numClasses;

for i = 1 : numLayers
    layerProp(i).activation = zeros(nInputs,1);
    if i<numLayers
        layerProp(i).net = zeros(neuronPerLayer(i),1);
        if i < numLayers-1
            layerProp(i).weights = -0.01*randn(nInputs, neuronPerLayer(i)+1);
            layerProp(i).weightsinc = zeros(nInputs, neuronPerLayer(i)+1);
            layerProp(i).bias = zeros(1,neuronPerLayer(i)+1);
            layerProp(i).biasinc = zeros(1,neuronPerLayer(i)+1);
        else
            layerProp(i).weights = -0.01*randn(nInputs, neuronPerLayer(i));
            layerProp(i).weightsinc = zeros(nInputs, neuronPerLayer(i));
            layerProp(i).bias = zeros(1,neuronPerLayer(i));
            layerProp(i).biasinc = zeros(1,neuronPerLayer(i));
        end
        layerProp(i).trainEpochs = trainEpochs;
        layerProp(i).LR = LR;
        layerProp(i).LRcurrent = layerProp.LR;
        layerProp(i).Mom = Mom;
        layerProp(i).Momcurrent = Mom;
        nInputs = neuronPerLayer(i);
    end
end

varargout = {layerProp};
varargout = varargout(1:nargout);
end