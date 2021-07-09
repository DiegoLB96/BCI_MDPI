function varargout = rbmBestSegment(Data, Targets, layerProp, windowSize, stepSize, fs, calcError)

if nargin < 4, windowSize = 1; end
if nargin < 5, stepSize = 1; end
if nargin < 6, fs = 1; end
if nargin < 7, calcError = 1; end
if size(calcError,2) ~= 2, calcError(2) = 4; end

testData = [];
testTargets = [];
trainData = Data{1};
trainTargets = Targets{1};

if calcError(1) && size(Data,2) > 1 && size(Targets,2) > 1 
    testData = Data{2};
    testTargets = Targets{2};
end

step = ceil( fs * stepSize ) - 1;                 
window = ceil( fs * windowSize ) - 1;               
% noverlap = window - step;
numSteps = floor(size(trainData,2)/step) - 1;

bestSegment = 0;
nextStep = 1;
acurracy = zeros(numSteps, 1);
tempAccuracy = 0;
tempLayerProp = layerProp;
tempErrRBM = 0;
tempErrBP = 0;

for j = 1: numSteps
    tempTrainData    = trainData(:, nextStep: nextStep+window, :);
    tempTrainData    = reshapeVector(tempTrainData);
    tempTrainTargets = trainTargets(:, nextStep: nextStep+window, :);
    tempTrainTargets = reshapeVector(tempTrainTargets);

 
    %% NORMALIZE DATA
%     totMean = mean(mean(tempTrainData,2));
%     totStd = 2*(mean(std(tempTrainData,0,2)));
    
%     tempTrainData = tempTrainData-totMean;
%     tempTrainData =  tempTrainData./totStd;
    
    %%%% Arrange data in Batches %%%
    batchsize = 2^(floor((nextpow2(size(tempTrainData,1))-2)/2));
    [tempTrainData, tempTrainTargets]   = makeBatches(tempTrainData, tempTrainTargets, batchsize); % arrange data in Batches
    
    if calcError(1) && ~isempty(testData) && ~isempty(testTargets) 
        tempTestData    = testData(:, nextStep: nextStep+window, :);
        tempTestData    = reshapeVector(tempTestData);
        tempTestTargets = testTargets(:, nextStep: nextStep+window, :);
        tempTestTargets = reshapeVector(tempTestTargets);
        
%         tempTestData = tempTestData-totMean;
%         tempTestData = tempTestData./totStd;
        
        [tempTestData, tempTestTargets]     = makeBatches(tempTestData, tempTestTargets, batchsize); % arrange data in Batches
    else
        tempTestData = [];
        tempTestTargets = [];
    end
     
    Data = {tempTrainData, tempTestData};
    Targets = {tempTrainTargets, tempTestTargets};
    
    
    %%%%% LEARNING %%%%%%
    [layerPropOut, ErrRBM, ErrBP] = RBM(Data, layerProp, Targets, [], calcError, 0);
    
%     if stepSize * j + 2 == 4
%        hshsh = 1;
%     end         
%     
    if ~isempty(tempTestData)
        acurracy(j) = max(1-ErrRBM.testErr);
    else
        acurracy(j) = max(1-ErrRBM.trainErr);
    end
    
    if acurracy(j) > tempAccuracy
        bestSegment = stepSize * j;
        tempAccuracy = acurracy(j);
        tempLayerProp = layerPropOut;
        tempErrRBM = ErrRBM;
        tempErrBP = ErrBP;
    end
    nextStep = nextStep + step;
end
   
varargout = {tempLayerProp, bestSegment, tempErrRBM, tempErrBP, acurracy};
varargout = varargout(1:nargout);
end