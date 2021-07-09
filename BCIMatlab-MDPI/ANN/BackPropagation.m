function [varargout] = BackPropagation(Data, layerProp, Targets, dmse, calcError, Echo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas
%   created 24.10.2010 - last modified 03.09.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main file to train BackPropagation
% INIT LAYERS
% Initilize a structure for the BackPropagation
%
% SYNTAX
% [layerProp, ErrRBMvarargout] = BackPropagation(Data, layerProp, Targets, ECHO)
% DEFINITION
% Creates a structure with N-number of hidden layers for the BP, structure
% needed for BP train
%
% %% activation %%
% Data: Cell array with the training data set(could include testing data set)
% layerProp: The initialized definition of the layer for the BP
% Targets: Cell array with the targets(or classes) for the training data
%           set(could include testing targets)
% %% net %%
% layerProp: structure with the train RBM layers containing all its parameters
% ErrRBM: 


plotcount = 0;
mse = Inf; % assuming worst classification

numLayers = size(layerProp,2);
maxiter = 1;
mm = 0;
fig = 2;
figure(fig)
testData = Data{2};
trainData = Data{1};
testTargets = Targets{2};
trainTargets = Targets{1};

trainTime = zeros(layerProp(1).trainEpochs, 1);

% while min(mse) > dmse && mm < maxiter
%     mm = mm + 1;
    for epoch = 1 : layerProp(1).trainEpochs
        mse = 0;
        tmse = 0;
        
        %%%%%%% COMPUTE ERROR %%%%%%%%%
        %%% Training Error %%%%
        if calcError
            [trainClassErr(epoch), trainRegErr(epoch)] = calcErrors(trainData, trainTargets, layerProp);
            
            %%% Testing Error %%%%
            [testClassErr(epoch), testRegErr(epoch)] = calcErrors(testData, testTargets, layerProp);
        end
        %%%%%%% END COMPUTE ERROR %%%%%%%%%
        tic
        for batch = 1:size(trainData,1)
            data = squeeze(trainData(batch,:,:))';
            target = squeeze(trainTargets(batch,:,:))';
            numPatterns = size(data,1);
            numTargets = size(target,2);
            
            %%%% FORWARD PROPAGATION %%%%%
            layerProp = forward(data, layerProp);
            % Error signal calculation training
            err = -(target'-layerProp(end).activation);% e(:,numPa) = target(:,numPa)-layerProp(end).net;
            % sse = sum(sum(err.^2)); % sum square error (Energy error calculation)

            % Calculation of Back propagation using the pattern numPa
            layerProp = backward(layerProp, err, numPatterns);
        end
        trainTime(epoch, 1) = toc;
        %         epochs = epochs + 1;
        if calcError && Echo && mod(epoch,20) == 0
            plotcount = plotcount + 1;
            set(0,'CurrentFigure',fig)
            plot(plotcount,trainClassErr(epoch),'ro')
            hold on
            plot(plotcount,testClassErr(epoch),'bo')
            hold on
        end
    end
% end
Err = struct('type', 'BP Training Error', ...
    'testRegErr', testRegErr, 'trainRegErr', trainRegErr,...
    'testClassErr', testClassErr, 'trainClassErr', trainClassErr);

varargout = {layerProp,Err, trainTime};
varargout = varargout(1:nargout);
end
