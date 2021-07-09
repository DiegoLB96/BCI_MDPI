function [varargout] = rbmTrain(epochs, Data, layerProp, Targets, ECHO)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas
%   created 24.10.2010 - last modified 03.06.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main file to train the individual layers of the RBMs
% INIT LAYERS
% Initilize a structure for the RBM
%
% SYNTAX
% [layerProp, batchdataout, errsum] = rbmTrain(epochs, Data, Targets, layerProp, ECHO)
% DEFINITION
% This program trains Restricted Boltzmann Machine in which visible, 
% stochastic inputs are connected to hidden, stochastic feature detectors 
% using symmetrically.
%
% %% Input %%
% epochs: gives the number of epoch in the training (could be used to have
%       a better update in adaptive)
% Data: Array with the input data with size(nTrials nInputs nbt), being
%       ntrials = number of trials, nInputs=number of features or inputs
%       and nbt: number of batches
% layerProp: initialized layer with all its properties(recomend using
%       rbmInitlayer)
% Targets: Array with the target or classes, with size(nTrials nClasses
%       nbt), with nTrials and nbt with the same size as Data and nClasses
%       in the form of oneRepresentation(see oneRepClasses in ./Extras)
% 
% %% Outputs %%
% layerProp: Returns the train RBM layer with all its properties
% batchdataout: is the data from wich a consecutive layer could be trained 
% errsum: is an internal reconstruction error of the layer after gibbs 
%       sampling 
% 


if nargin < 4, Targets = []; end
if nargin < 5, ECHO = 0; end

trainPorcentage = epochs/layerProp.trainEpochs;

[nTrials nInputs nbt] = size(Data); % nTrials: number of trials the batch contains, nbt: number of batches per trial
errsum=0; 
batchdataout = zeros(nTrials, layerProp.numNeurons, nbt);
for batch = 1:nbt,
    if ECHO, fprintf(1,'batch %d\r',batch); end
    data = Data(:,:,batch);
    classes = Targets(:,:,batch);
    
    %%%% CONSTRASTIVE DIVERGANGE %%%%
    if  sum(strcmpi(layerProp.trainingType,{'RBM','RBMClass','RBMNN','RBMHybrid'}))>0
        [dw, dhb, dvb, dcw, dcb, outData, err] = rbmConstrastiveD(data, layerProp, classes);

        %%% Reconstruction error %%%
        errsum = err + errsum;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ~strcmpi(layerProp.trainingType,'RBMHybrid')
        Delta = {dw, dhb, dvb, dcw, dcb};
        [layerProp] = rbmConstrastiveDUpdate(trainPorcentage, layerProp, Delta, nTrials);
        batchdataout(:,:,batch)  = outData;
        end
    end
    %%%% END CONSTRASTIVE DIVERGANGE %%%%
    %%%% DISCRIMINATIVE TRAINING %%%%
    if  sum(strcmpi(layerProp.trainingType,{'RBMdis','RBMHybrid'}))>0
        if ~isempty(classes) && ~isempty(layerProp.classWeights)
            [layerProp] = rbmDiscriminative(data, layerProp, classes, trainPorcentage);
        else
            disp('Cannot do discriminative')
        end
    end
    %%%% END DISCRIMINATIVE TRAINING %%%%
end

varargout = {layerProp, batchdataout, errsum};
varargout = varargout(1:nargout);

end

