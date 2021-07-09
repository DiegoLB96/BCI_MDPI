function varargout = rbmConstrastiveD(data, layerProp, classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   03.12.2010-27.01.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBM DEFINITIONS
%
% SYNTAX
% [dvw, dhb, dvb, dcw, dcb, outData, err] = rbmConstrastiveD(data, layerProp, visFunct, ngibbs, classes, classLayerProp)
% does Constrastive Divergance
% 
% DEFINITION
% Does Constrastive Divergance(CD) with the resulting in the updates of the weights
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

if nargin < 2, error('Not enough input arguments'); end
if nargin < 3 || isempty(classes), classes = 0; end
if ~sum(strcmpi(layerProp.trainingType, {'RBMclass','RBMdis','RBMHybrid'}))
    classWeights = 0;               % weights of the classification layer
    classBiases = 0;                % biases of the classification layer
    classFunct = struct([]);        % activation functions of the classification layer
    classes = 0;
else
    classWeights = layerProp.classWeights;
    classBiases = layerProp.classBiases;
    classFunct = layerProp.classFunct;
end

hidFunct  = layerProp.hidFunct;     %structure with the specified reaction,  
visFunct  = layerProp.visFunct;     %state and input functions for the different RBM
weights = layerProp.weights;        % weights of the current Hidden unit
hidbiases = layerProp.hidbiases;    % hidden biases for the current RBM
visbiases = layerProp.visbiases;    % visible biases for the current RBM

nClasses = size(classes,2);

ngibbs =  layerProp.ngibbs;

% default class biases and weight updates
dcw = []; % class weights
dcb = []; % class bias


%%%%%%%%% POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    IHidden = hidFunct.Input(weights,hidbiases,data,classes,classWeights); % Input for hidden units %
    RHidden = hidFunct.React(IHidden);  % Reaction of hidden units %
    SHidden = hidFunct.State(RHidden,IHidden); % States of hidden units %
    %%% Positive Phase statistics
    outData     = RHidden;
    posprods    = data' * RHidden;
	posCprods	= RHidden'*classes;
    poshidact   = sum(RHidden,1);
    posvisact   = sum(data,1);
	if nClasses > 1, posCact = sum(classes,1); end

%%%%%%%%% Markov Chain (Gibbs sampling)  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ng = 1 : ngibbs
        %%% Sample hidden states
        IVisible = visFunct.Input(weights',visbiases,SHidden);% Input for visible units %
		if nClasses > 1, IClass = classFunct.Input(classWeights,classBiases,SHidden); end  
		RVisible = visFunct.React(IVisible); %%%% NEGDATA
		if nClasses > 1, RClass = classFunct.React(IClass); else RClass = zeros(size(classes)); end 
        SVisible = visFunct.State(RVisible,IVisible); % States of visible units %
		if nClasses > 1, SClass = classFunct.State(RClass,IClass); end 

        %%%%%%%%% NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        InHidden = hidFunct.Input(weights,hidbiases,RVisible,RClass,classWeights);
        RnHidden = hidFunct.React(InHidden);
        if ng < ngibbs, SHidden = hidFunct.State(RnHidden,InHidden); end
    end
%%% Negative Phase statistics
    negprods  = RVisible'*RnHidden;
    neghidact = sum(RnHidden,1);
    negvisact = sum(RVisible,1);
	if nClasses > 1, 
		negCact = sum(RClass,1);
		negCprods = RnHidden'*RClass;
	end 
%%% Sum up statistics
    dw = (posprods - negprods);    % Layer weights
    dvb = (posvisact - negvisact);  % Visible biases
    dhb = (poshidact - neghidact);  % Hidden biases
	if nClasses > 1, 
		dcw = (posCprods - negCprods);
		dcb = (posCact - negCact);
	end 

% ERROR %    
err = sum(sum( (data - RVisible).^2 )); % reconstruction error of current minibatch
    
varargout = {dw, dhb, dvb, dcw, dcb, outData, err};
varargout = varargout(1:nargout);
end
