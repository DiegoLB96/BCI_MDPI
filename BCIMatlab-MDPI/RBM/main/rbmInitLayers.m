function [varargout] = rbmInitLayers(nInputs, trainingType, neuronPerLayer, typeVisible, typeLayer, trainEpochs, ...
    LR, CT, MOM, ngibbsLayer, ...
    nClasses, typeClass, LRc, CTc, MOMc,...
    totStd, totMean)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INIT LAYERS
% Initilize a structure for the RBM 
%
% SYNTAX
% [layerProp] = rbmInitLayers(nInputs, trainingType, neuronPerLayer, typeVisible, typeLayer, ...
%     LR, CT, MOM, ngibbsLayer, ...
%     typeClass, nClasses, LRc, CTc, MOMc,...
%     totStd)
% 
% DEFINITION
% Creates a structure with N-number of hidden layers for the RBM, structure
% needed for RBM train
% 
% %% Inputs %%
% nInputs: Specify the number of neurons needed as Input vector(Real value)
% trainingType: String array with the form of train for the RBM 
%           RBMNN: neural network, trains the RBM layers with out of the
%               information of the class using Constrastive Divergance and 
%               then insert the last layer and train it as a Neural Network 
%               with backpropagation 
%           RBMclass: RBM Using Classifier, Uses the information of the
%               class in the last layer as part of the input to maximize the
%               likelihood of the parameters
%           RBMdis: RBM using Discriminative, The same form as RBMClass,
%               but uses the Discriminative training for the last layer
%           RBMHybrid: Uses RBM hybrid with generative and discriminative
%               training
%           TODO...
%           RBMSemi: Semi-supervised training of RBM
%           RBMDeep: Could be use for deep auto encoding training, it uses
%               backpropagation to train the unfolded RBMeverything is 
%               already implemented just need to code it correctly
%           
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
% MOM: specify the Momentum Has to be inserted in a cell as numeric or as a function 
%               See rbmConvertRates 
% ngibbsLayer: tells the gibbs sample each layer is going to be trained
% typeClass: string with the form of the first layer(Visible layer)
%             [Binary,Rlin, Gauss, Softmax, noState], if empty or or not specified
%             it fills with the Softmax unit
% nClasses: Specify the number of classes(Real value) if not specified it
%             uses zero and doesnt use the class for classification
% LRc, CTc and MOMc: the same as previous learning parameters but for the
%           class
% totStd: tells the standard deviation of the class (Not fully implemented)
% nClasses: number of output classes for classificatioTRangen
%
% %% Outputs %%
% layerProp: structure with the initilized layers parameters
% layerProp.numNeurons: number of Neurons of Hidden layer
%           typeHidden: type of hidden layer
%           typeVisible: type of visible layer
%           trainingType: training type
%           hidFunct: Activation function of hidden layer
%           visFunct: Activation function of visible layer
%           weights: symetric weights between visible and hidden layer
%           hidbiases: biases of hidden layer
%           visbiases: biases of visible layer
%           weightsinc: current increments of weights
%           hidbiasinc: current increments of hidden biases
%           visbiasinc: current increments of visible biases
%           epsilonw: learning rate of weights
%           epsilonvb: learning rate of visible biases
%           epsilonhb: learning rate of hidden biases
%           costw: Cost of weights
%           costvb: Cost of visible biases
%           costhb: Cost of hidden biases
%           momentumw: Momentum of weights
%           momentumvb: Momentum of visible biases
%           momentumhb: Momentum of hidden biases
%           ngibbs: number of gibbs sample of layer
%           totStd: Standar deviation of previous data(Not currently implemented)
%           trainEpochs: Number of training epochs of current layer
%           typeClass: type of activation function of class If available 
%           numClasses: number of classes of the current layer. If available
%           classFunct: activation function structure of the classes
%           classWeights: symetric weights between hidden and class layers
%           classBiases: biases of the class layer
%           epsiloncw: learning rate for the weights between class and hidden layers
%           epsiloncb: learning rate for the biases of the class layer
%           costcw: cost for the weights between class and hidden layers
%           costcb: cost of biases of class layer
%           momentumcw: momentum for the weights between class and hidden layers
%           momentumcb: momentum for the biases of the class layer

if nargin < 2, error('Needs at least the training type and number of features'); end
if nargin < 3, 
    neuronPerLayer = 50; 
elseif isempty(neuronPerLayer),  
    nNeurons = nextpow2(nInputs); 
    neuronPerLayer = [2^(nNeurons)]; 
    if neuronPerLayer*.7 < nInputs, neuronPerLayer = neuronPerLayer * 2; end
end
if nargin < 4, typeVisible = 'Binary'; end
if nargin < 5, typeLayer = cell(1,size(neuronPerLayer,2)); end
if nargin < 6, trainEpochs = []; end
if isempty(trainEpochs), trainEpochs = 100; end
if nargin < 7, LR = {}; end % learning rates
if nargin < 8, CT = {}; end % costs
if nargin < 9, MOM = {}; end % momentums
if nargin < 10, ngibbsLayer = []; end
if nargin < 11, nClasses = 0; end
if nargin < 12, typeClass = 'Softmax'; end
if nargin < 13, LRc = {}; end
if nargin < 14, CTc = {}; end
if nargin < 15, MOMc = {}; end
if nargin < 16, totStd = 1; end
if nargin < 17, totMean = 0; end


numLayers = size(neuronPerLayer,2);

typeLayerTemp = cell(1,numLayers);
for i = 1:numLayers
    if i > size(typeLayer,2) || isempty(typeLayer{1,i}) || ~sum(strcmpi(typeLayer{1,i}, {'Binary', 'Rlin', 'Gauss', 'Softmax', 'noState','Tansig'}))
        typeLayerTemp{1,i} = 'Binary';
        warning(['Layer: ' num2str(i) ' was not filled or with not a defined type.'...
            'It will be replaced by Binary'])
    else
        typeLayerTemp{1,i} = typeLayer{1,i};
    end
    
    if isempty(ngibbsLayer), ngibbsLayer = 1; end
    if i > size(ngibbsLayer,2)
        ngibbsLayer(1,i) = ngibbsLayer(1,i-1);
    end
end
typeLayer = typeLayerTemp;

if size(trainEpochs,1) < numLayers,
   j =  size(trainEpochs,1);
   for i = j+1 : numLayers
       trainEpochs(i) = trainEpochs(i-1);
   end
end

if size(totStd,1) < numLayers,
   j =  size(totStd,1);
   for i = j+1 : numLayers
       totStd(i) = 1; % fill with ones (could have better implementation) 
   end
end

if size(totMean,1) < numLayers,
   j =  size(totMean,1);
   for i = j+1 : numLayers
       totMean(i) = 0; % fill with zeros (could have a better implementation) 
   end
end

% if nClasses ~= 0 && strcmpi(trainingType, 'RBMclass')
%     numLayers = numLayers + 1; 
%     neuronPerLayer = [neuronPerLayer nClasses];
%     typeLayer{end+1} = 'Softmax';
% end

s = struct('numNeurons',0, 'typeHidden','', 'typeVisible','','trainingType','',...
        'hidFunct',struct([]), 'visFunct', struct([]),...
        'weights',[],'hidbiases',[],'visbiases',[],...
        'weightsinc',[],'hidbiasinc',[],'visbiasinc',[],...
        'epsilonw',[],'epsilonvb',[],'epsilonhb',[],...
        'costw', [], 'costvb', [], 'costhb', [],...
        'momentumw', [], 'momentumvb', [], 'momentumhb', [],...
        'ngibbs',1,...%         
        'totStd',1, 'totMean', 0,...
        'trainEpochs', 0,...
        'typeClass','','numClasses',[],...
        'classFunct',struct([]),...
        'classWeights',[],'classBiases',[],...
        'epsiloncw',[],'epsiloncb',[],...
        'costcw', [], 'costcb', [],...
        'momentumcw', [], 'momentumcb', []);
layerProp = repmat(s,1,numLayers);
insertClass = 0;
if nClasses ~= 0 && sum(strcmpi(trainingType, {'RBMclass','RBMNN','RBMdis','RBMHybrid'}))
    insertClass = 1;
end

TV = typeVisible;    
%%%%%%%% FILL IN THE STRUCTURE %%%%%%%%
for i = 1 : numLayers
    numhid = neuronPerLayer(i);
    layerProp(i).ngibbs = ngibbsLayer(i);
    layerProp(i).numNeurons = numhid; 
    layerProp(i).typeHidden  = typeLayer{i};
    layerProp(i).typeVisible  = TV;
    layerProp(i).trainEpochs = trainEpochs(i);
    layerProp(i).totStd = totStd(i);
    layerProp(i).totMean = totMean(i);
%     if strcmpi(TV,'Gauss') && exist('totStd','var'), layerProp(i).totStd = totStd; end
    
    if insertClass && i == numLayers
        layerProp(i).typeClass  = typeClass;
        layerProp(i).numClasses = nClasses;
        layerProp(i).trainingType = trainingType;
    else
        layerProp(i).trainingType = 'RBM'; % TODO: Change if a different way of training is used for inner layers
    end
    
    %%% INITIALIZING SYMETRIC WEIGHTS AND BIASES %%%
    switch typeLayer{1,i}
        case 'Binary'
            layerProp(i).weights    = 0.1*randn(nInputs, numhid);               % Weights
            layerProp(i).hidbiases  = zeros(1,numhid);                          % Hidden bias, 0.1*randn(1,numhid);%
            layerProp(i).visbiases  = zeros(1,nInputs);                         % Visible bias, 0.1*randn(1,nInputs);%
        case 'Gauss'
            layerProp(i).weights    = 0.1*randn(nInputs, numhid);              
            layerProp(i).hidbiases  = zeros(1,numhid);                          
            layerProp(i).visbiases  = zeros(1,nInputs);                         
        case 'Rlin' % TODO: dont know if this is correct!!!!
            layerProp(i).weights    = 0.01*randn(nInputs, numhid);%./(totStd/4);%/sqrt(numhid);              
            layerProp(i).hidbiases  =  -1*(1:numhid)+.5;%1/numhid*(1:numhid);%
            layerProp(i).visbiases  = zeros(1,nInputs);
        case 'Softmax'
            layerProp(i).weights    = 0.01*randn(nInputs, numhid);               
            layerProp(i).hidbiases  = zeros(1,numhid); 
            layerProp(i).visbiases  = zeros(1,nInputs);
        otherwise
            layerProp(i).weights    = 0.01*randn(nInputs, numhid);               
            layerProp(i).hidbiases  = zeros(1,numhid); 
            layerProp(i).visbiases  = zeros(1,nInputs);
    end
    	% Internal matrices    
%    layerProp(i).RHidden = zeros(nTrials,numhid);           % Positive hidden Response
%    layerProp(i).RnHidden = zeros(nTrials,numhid);          % Negative hidden Response
    layerProp(i).weightsinc = zeros(nInputs,numhid);         % Weights increment
    layerProp(i).hidbiasinc = zeros(1,numhid);               % Hidden Bias increment
    layerProp(i).visbiasinc = zeros(1,nInputs);              % Visible Bias increment
    
    if insertClass && i == numLayers
        switch typeClass%%%% not compleatly implemented at the moment
            case 'Binary'
                layerProp(i).classWeights = 0.1*randn(numhid, nClasses);
                layerProp(i).classBiases  = zeros(1,nClasses);                %%% Class visible biases
            case 'Gauss'
                layerProp(i).classWeights = 0.1*randn(numhid, nClasses);
                layerProp(i).classBiases  = zeros(1,nClasses);
            case 'Rlin' % TODO: dont know if this is correct!!!!
                layerProp(i).classWeights = 0.1*randn(numhid, nClasses)/sqrt(nClasses);%1./(totStd*ones(numhid, nClasses)); 
                layerProp(i).classBiases  = zeros(1,nClasses);%-.5:-1:-(nClasses-.5);
            case 'Softmax'
                layerProp(i).classWeights = 0.01*randn(numhid, nClasses);
                layerProp(i).classBiases  = zeros(1,nClasses);
            otherwise
                layerProp(i).classWeights = 0.1*randn(numhid, nClasses);
                layerProp(i).classBiases  = zeros(1,nClasses);
        end
        layerProp(i).classWeightsinc = zeros(numhid,nClasses);         % Weights increment
        layerProp(i).classBiasesinc = zeros(1,nClasses);               % Hidden Bias increment
    end
    
%     layerProp(i).batchdataout = zeros(nTrials,numhid,nbt); % Batch data for the next layer
    
    [LR, CT, MOM] = rbmConvertRates(LR, CT, MOM, typeLayer, typeVisible, 3);
    %%%% LEARNING RATES, COSTS AND MOMENTUMS %%%% 
    % Learning rate
    layerProp(i).epsilonw      = LR{i,1};   % weights
    layerProp(i).epsilonvb     = LR{i,2};   % biases visible units
    layerProp(i).epsilonhb     = LR{i,3};   % biases hidden units
    % Costs: penalization to avoid large weights
    layerProp(i).costw    	= CT{i,1};
    layerProp(i).costvb    	= CT{i,2};
    layerProp(i).costhb    	= CT{i,3};
    % Momentums
    layerProp(i).momentumw   = MOM{i,1};
    layerProp(i).momentumvb  = MOM{i,2};
    layerProp(i).momentumhb  = MOM{i,3};
    
    %%%%% Restricted Boltzmann Machines Transfer Function Definitions %%%%%%
    [layerProp(i).hidFunct, layerProp(i).visFunct] = rbmDefinitions(typeLayer{1,i}, TV);
    
    if insertClass && i == numLayers
        [LRc, CTc, MOMc] = rbmConvertRates(LRc, CTc, MOMc, typeClass, typeLayer{1,end}, 2);
        % Learning rate
        layerProp(i).epsiloncw = LRc{1,1};   % weights
        layerProp(i).epsiloncb = LRc{1,2};   % biases hidden units
        % Costs: penalization to avoid large weights
        layerProp(i).costcw = CTc{1,1};
        layerProp(i).costcb = CTc{1,2};
        % Momentums
        layerProp(i).momentumcw  = MOMc{1,1};
        layerProp(i).momentumcb  = MOMc{1,2};
        
        %%%%% Restricted Boltzmann Machines Transfer Function Definitions %%%%%%
        [~, layerProp(i).classFunct] = rbmDefinitions(typeLayer{1,i}, typeClass);
    end
    
    TV = typeLayer{i};
    nInputs = numhid;
end

varargout = {layerProp};
varargout = varargout(1:nargout);


end
