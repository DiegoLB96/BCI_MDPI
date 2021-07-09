function [varargout] = RBM(Data, layerProp, Targets, maxepochBack, calcError, ECHO)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas
%   created 24.10.2010 - last modified 03.09.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main file to train RBMs
% INIT LAYERS
% Initilize a structure for the RBM
%
% SYNTAX
% [layerProp, ErrRBM] = RBM(Data, layerProp, Targets, calcError,  ECHO);
% DEFINITION
% Creates a structure with N-number of hidden layers for the RBM, structure
% needed for RBM train
%
% %% Inputs %%
% Data: Cell array with the training data set(could include testing data set)
% layerProp: The initialized definition of the layer for the RBM
% Targets: Cell array with the targets(or classes) for the training data
%           set(could include testing targets)
% maxepochBack: max epochs for back propagation, normally not used
% calcError: Array with first element being a boolean that defines if the
%               error is going to be calculated and the second being the 
%               number of segments to calculate the error this to reduce
%               the training interval
% ECHO: It defines if the program would return all the steps followed in
%       the training
%
% %% Outputs %%
% layerProp: structure with the train RBM layers containing all its parameters
% ErrRBM: Error structure filled with all the posible error calculated
%       testErr: classification error of the testing data set
%       testCrerr:
%       testNumClass: Return the class obtained while calculating the
%                   testErr
%       testRecErr: Error in reconstruction of the testing data set
%       trainErr: classification error of the training data set
%       trainCrerr:
%       trainNumClas: Return the class obtained while calculating the
%                   trainErr
%       trainRecErr: Error in reconstruction of the training data set

if nargin < 2, error('Insert initialized Layer properties'); end
if nargin < 3,
    warning('WarnTests:convertTest','The Targets were not defined it \n would use an empty array');
    Targets = {[]};
end
if nargin < 4 || isempty(maxepochBack), maxepochBack = 200; end
if nargin < 5, calcError = 0; end
if size(calcError,2) < 2, calcError(2) = 1;end
if nargin < 6, ECHO = 0; end

numLayers = size(layerProp,2);

testData = [];
testTargets = [];
trainData = Data{1};
trainTargets = Targets{1};
tempData = trainData;
tempTargets = trainTargets;

if calcError(1) && size(Data,2) > 1 && size(Targets,2) > 1
    testData = Data{2};
    testTargets = Targets{2};
end

[nTrials, nInputs, nbt] = size(trainData); % nbt: number of trials per batch

%%% To calculate the error %%%
sizeErr      = floor(layerProp(end).trainEpochs/calcError(2));
if calcError(2) ~= 1, sizeErr = sizeErr + 1; end
testErr      = zeros(sizeErr, 1);
testCrerr    = zeros(sizeErr, 1);
testNumClass = zeros(sizeErr, size(trainTargets,2));
testRecErr   = zeros(sizeErr, numLayers);
trainErr	 = zeros(sizeErr, 1);
trainCrerr   = zeros(sizeErr, 1);
trainNumClass = zeros(sizeErr, size(trainTargets,2));
trainRecErr  = zeros(sizeErr, numLayers);

trainTime = zeros(layerProp.trainEpochs,numLayers);

for n = 1 : numLayers
    ttError = 0;
    numNeurons = layerProp(n).numNeurons;
    if ECHO, fprintf(1,'Pretraining Layer %d with RBM: %d-%d \n', n , nInputs ,numNeurons ); end
    disp(['RBM: ' num2str(n) ' with the form ' layerProp(n).typeVisible ' - ' layerProp(n).typeHidden])
    
%     if n == numLayers, calcError(1) = 1; end
    
    epoch = 1; % TODO: change for different training parts(could be used in adaptive)
    errorLayerTemp = 1; %% used to get the best training weights %%%
    
    for epoch = epoch : layerProp(n).trainEpochs,
        %         if ~mod(epoch,2)
        %             plotStates(layerProp, trainData,1);  %pause
        %         end
        %%%%%%% COMPUTE ERROR %%%%%%%%%
        if ~mod(epoch, calcError(2)) || epoch == 1 % it calculates the error before training
            ttError = ttError + 1;
            %%%%%%% COMPUTE TRAINING ERROR %%%%%%%%%
            if calcError(1) && ~isempty(trainData)
                if ~isempty(trainTargets) && n == numLayers && ~strcmpi(layerProp(1,n).trainingType,'RBMDeep')
                    [trainErr(ttError), trainCrerr(ttError) trainNumClass(ttError,:)] = rbmClasificationErrorGibbsSamp(trainData, trainTargets, layerProp);
                end
                trainRecErr(ttError,n) = rbmReconstructionError(trainData, layerProp, n);
            end
            %%%%%%% COMPUTE TEST ERROR %%%%%%%%%
            if calcError(1) && ~isempty(testData)
                if ~isempty(testTargets) && n == numLayers && ~strcmpi(layerProp(1,n).trainingType,'RBMDeep')
                    [testErr(ttError), testCrerr(ttError) testNumClass(ttError,:)] = rbmClasificationErrorGibbsSamp(testData, testTargets, layerProp);
                end
                testRecErr(ttError,n) = rbmReconstructionError(testData, layerProp, n);
            end
        end
        %%%%%%% END COMPUTE ERROR %%%%%%%%%
        
        %%%%%%% START RBM TRAINING %%%%%%%%
        tic
        [layerProp(n), batchdataout, errsum] = rbmTrain(epoch, tempData, layerProp(n), tempTargets, ECHO);
        trainTime(epoch, n) = toc;
        %%%%%%% END RBM TRAINING %%%%%%%%
        
        if ECHO,
            fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
            plot(epoch,errsum); hold on; drawnow
        end
        
        %%% Take the layer with the best properties %%%
        if calcError(1) && ~mod(epoch, calcError(2)), % if the error is not calculated then it takes the last one in the training
            if  epoch > 10 % TODO has to be optimized for the different types of error %%
                if testErr(ttError) == 0, errTemp = testRecErr(ttError,n); else errTemp = testErr(ttError); end
                if errorLayerTemp > errTemp
                    errorLayerTemp = errTemp;
                    layerPropTemp = layerProp(n);
                    %                 disp(num2str(epoch))
                end
            end
        else
            layerPropTemp = layerProp(n);
        end
    end
    layerProp(n) = layerPropTemp;
    tempData = batchdataout;
    nInputs = numNeurons;
end
Err = struct('type', 'RBM Training Error', 'testErr', testErr, 'testCrerr', testCrerr, 'testNumClass', testNumClass,'testRecErr',testRecErr,...
    'trainErr', trainErr, 'trainCrerr',trainCrerr, 'trainNumClass', trainNumClass, 'trainRecErr',trainRecErr);

trainTimeBp = 0;
%%%%% BACK PROPAGATION %%%%%
if sum(strcmpi(layerProp(1,end).trainingType,{'RBMNN'} ))
    disp('Back Propagation')
    tic
    [layerProp, ErrBP] = backProp(Data, Targets, layerProp, maxepochBack, calcError, 0);
    trainTimeBp = toc;
else
    ErrBP = struct('type', 'Back Propagation Error', 'testErr', [], 'testCrerr', [], 'trainErr', [], 'trainCrerr',[],...
        'testNumClass',[],'trainNumClass',[], 'testRecErr',[], 'trainRecErr',[]);
end

varargout = {layerProp, Err, ErrBP, trainTime, trainTimeBp};
varargout = varargout(1:nargout);
end
