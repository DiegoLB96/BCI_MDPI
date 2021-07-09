function [varargout] = SOM(Data, layerProp, I, J, Echo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas
%   created 24.10.2010 - last modified 03.09.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main file to train SOM
% INIT LAYERS
% Initilize a structure for the SOM
%
% SYNTAX
% [layerProp, ErrRBMvarargout] = SOM(Data, layerProp, Targets, maxepochBack, calcError, ECHO)
% DEFINITION
% Creates a structure with layers for the SOM,
%
% %% Inputs %%
% Data: Cell array with the training data set(could include testing data set)
% layerProp: The initialized definition of the layer for the RBM
% Targets: Cell array with the targets(or classes) for the training data
%           set(could include testing targets)
% calcError: Array with first element being a boolean that defines if the
%               error is going to be calculated and the second being the
%               number of segments to calculate the error this to reduce
%               the training interval
% %% Outputs %%
% layerProp: structure with the train SOM layers containing all its parameters
% ErrRBM:

[nTrials, nInputs, nbt] = size(Data); % nbt: number of trials per batch

plotcount = 0;
trainTime = zeros(layerProp.trainEpochs,1);

for epoch = 1 : layerProp.trainEpochs
    tic
    for nT = 1:nTrials
        for nB = 1:nbt
            xn = Data(nT,:,nB);
            % find the winner Neuron
            ind = somWinnerNeuron (xn,layerProp.weights);
            %// the 2-D index
            ri = [I(ind), J(ind)];
            %// distance between this node and the winner node.
            distance = ([I( : ), J( : )] - repmat(ri, size(layerProp.weights,1),1));
            distance = sqrt(sum(distance.^2,2));
            theta = exp(-distance.^2/(2*layerProp.sigmaActual.^2)); %
            theta = theta*layerProp.etaActual; % theta times learning rate!!
            
            %// updating weights
            for rr = 1:size(layerProp.weights,1)
                layerProp.weights(rr,:) = layerProp.weights(rr,:) + theta(rr).*( xn - layerProp.weights(rr,:));
            end
            
        end
    end
    
    %// update learning rate
    layerProp.etaActual = layerProp.eta * exp(-epoch/layerProp.tauEta);
    %// update sigma
    %sigN = sigN/2;
    layerProp.sigmaActual = layerProp.sigma*exp(-epoch/layerProp.tauSigma);
    
    trainTime(epoch, 1) = toc;
    if Echo
        plotSOM(layerProp.weights,xn)
        plotcount = 0;
    end
    plotcount = plotcount + 1;
end

varargout = {layerProp, trainTime};
varargout = varargout(1:nargout);
end
