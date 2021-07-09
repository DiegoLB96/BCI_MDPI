function [varargout] = rbmUpdateLayers(layerProp, trainEpochs,...
    LR, CT, MOM, ngibbsLayer, ...
    LRc, CTc, MOMc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 09.02.2011 - last modified 09.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INIT LAYERS
% Update the RBM structure for retrain
%
% SYNTAX
% [layerProp] = rbmInitLayers(trainEpochs, ...
%     LR, CT, MOM, ngibbsLayer, ...
%     typeClass, nClasses, LRc, CTc, MOMc,...
%     totStd)
% 
% DEFINITION
% Creates a structure with N-number of hidden layers for the RBM, structure
% needed for RBM train
% 
% %% Inputs %%
% layerProp: the RBM to be updated
% trainEpochs: the number of training epochs per layer
% LR: specify the learning rate. Has to be inserted in a cell as numeric or as a function 
%               See rbmConvertRates 
% CT: specify the costs. Has to be inserted in a cell as numeric or as a function 
%               See rbmConvertRates 
% MOM: specify the Momentum Has to be inserted in a cell as numeric or as a function 
%               See rbmConvertRates 
% ngibbsLayer: tells the gibbs sample each layer is going to be trained
% LRc, CTc and MOMc: the same as previous learning parameters but for the
%           class
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

if nargin < 2, trainEpochs = []; end
if isempty(trainEpochs), trainEpochs = 100; end
if nargin < 3, LR = {}; end % learning rates
if nargin < 4, CT = {}; end % costs
if nargin < 5, MOM = {}; end % momentums
if nargin < 6, ngibbsLayer = []; end
if nargin < 7, LRc = {}; end
if nargin < 8, CTc = {}; end
if nargin < 9, MOMc = {}; end



if size(trainEpochs,1) < numLayers,
   j =  size(trainEpochs,1);
   for k = j+1 : numLayers
       trainEpochs(k) = trainEpochs(k-1);
   end
end


%%%%%%%% FILL IN THE STRUCTURE %%%%%%%%
for j = 1 : numLayers
    layerProp(j).ngibbs = ngibbsLayer(j);
    layerProp(j).trainEpochs = trainEpochs(j);
   
    [LR, CT, MOM] = rbmConvertRates(LR, CT, MOM, typeLayer, typeVisible, 3);
    %%%% LEARNING RATES, COSTS AND MOMENTUMS %%%% 
    % Learning rate
    layerProp(j).epsilonw      = LR{j,1};   % weights
    layerProp(j).epsilonvb     = LR{j,2};   % biases visible units
    layerProp(j).epsilonhb     = LR{j,3};   % biases hidden units
    % Costs: penalization to avoid large weights
    layerProp(j).costw    	= CT{j,1};
    layerProp(j).costvb    	= CT{j,2};
    layerProp(j).costhb    	= CT{j,3};
    % Momentums
    layerProp(j).momentumw   = MOM{j,1};
    layerProp(j).momentumvb  = MOM{j,2};
    layerProp(j).momentumhb  = MOM{j,3};
    
    if insertClass && j == numLayers
        [LRc, CTc, MOMc] = rbmConvertRates(LRc, CTc, MOMc, typeClass, typeLayer{1,end}, 2);
        % Learning rate
        layerProp(j).epsiloncw = LRc{1,1};   % weights
        layerProp(j).epsiloncb = LRc{1,2};   % biases hidden units
        % Costs: penalization to avoid large weights
        layerProp(j).costcw = CTc{1,1};
        layerProp(j).costcb = CTc{1,2};
        % Momentums
        layerProp(j).momentumcw  = MOMc{1,1};
        layerProp(j).momentumcb  = MOMc{1,2};
    end

end

varargout = {layerProp};
varargout = varargout(1:nargout);


end
