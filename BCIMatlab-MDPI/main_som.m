% SOM CLASSIFIER 
%%%% SOM CLASSIFIER %%%%%%
% Training Data load
addpath(['Extras'],['SOM']);
subject = 1;                 % Subject Number
dataTransform = 'SFT'; % SFT, Wavelets, Time
[trainData, trainTargets, testData, testTargets, dataStats] = loadData();
parsave([rootpathResults 'Data' num2str(n) '.mat'], testData, testTargets, trainData, trainTargets, dataStats)
    
trainEpochs = 50; % number of epochs to train
LR = 0.2; % Learning rate eta
% Tau = [1000,1000];
Sigma = 0; % distance of radios of update
calcError = 1;

parfor n = stSubj:numSubj
    g = load([rootpathResults 'Data' num2str(n) '.mat'])
    trainData = g.trainData;
    trainTargets= g.trainTargets;
    testData = g.testData;
    testTargets = g.testTargets;
    dataStats = g.dataStats;
    numFeatures     = size(trainData, 2); % equal to the number of weights
    numClasses      = size(trainTargets, 2);
    neuronPerLayer = 100;%(sqrt(numFeatures)*2)^2;
    Sigma = floor(sqrt(neuronPerLayer));
%     Tau(2) = trainEpochs/log(Sigma);
    %%%%%%%%%% CLASIFIERS %%%%%%%%%%%%
    %%%%% INITIALIZE LAYERS %%%%%
    layerProp = somInitLayers(numFeatures, neuronPerLayer, trainEpochs, ...
            LR, Sigma);
    
    %%%% Arrange data in Batches %%%
    batchsize = 2^(floor((nextpow2(size(trainData,1))-2)/2));
    [trainData, trainTargets] = makeBatches(trainData, trainTargets, batchsize); % arrange data in Batches
    [testData, testTargets] = makeBatches(testData, testTargets, batchsize); % arrange data in Batches
    Data = {trainData, testData};
    Targets = {trainTargets, testTargets};
    
%     clear dataStats testData testTargets trainData trainTargets
    
    %%%%% LEARNING %%%%%%
    %//map index
    [I, J] = ind2sub([sqrt(neuronPerLayer), sqrt(neuronPerLayer)], 1:neuronPerLayer);
    % training
    [layerProp, trainTime] = SOM(Data{1}, layerProp, I, J, 1);
    
    parsave([rootpathResults 'SOMdata' num2str(n) '.mat'], trainTime, layerProp)
end


