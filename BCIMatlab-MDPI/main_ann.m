% Backpropagation Algorithm for written digits
% David Balderas
% TEC Monterrey
% Training Data load
addpath(['Extras'],['ANN']);
subject = num2str(1);                 % Subject Number
dataTransform = 'SFT'; % SFT, Wavelets, Time
[trainData, trainTargets, testData, testTargets, dataStats] = loadData(subject);
trainEpochs = 100; % number of epochs to train each layer of the RBM 
neuronPerLayer = [1000];

LR = 0.05; % Learning rate eta
% Tau = [1000,1000];
Mom = 0.05; % distance of radios of update
calcError = 1;

%  dbstop if naninf
parfor n = stSubj:numSubj
    g = load([rootpathResults 'Data' num2str(n) '.mat']);
    trainData = g.trainData;
    trainTargets= g.trainTargets;
    testData = g.testData;
    testTargets = g.testTargets;
    dataStats = g.dataStats;
    numFeatures     = size(trainData, 2); % equal to the number of weights
    numClasses      = size(trainTargets, 2);
%     Tau(2) = trainEpochs/log(Sigma);
    %%%%%%%%%% CLASIFIERS %%%%%%%%%%%%
    %%%%% INITIALIZE LAYERS %%%%%
    layerProp = BPInitLayers(numFeatures, neuronPerLayer, numClasses, trainEpochs, ...
        LR, Mom);
    
    %%%% Arrange data in Batches %%%
    batchsize = 2^(floor((nextpow2(size(trainData,1))-2)/2));
    [trainData, trainTargets] = makeBatches(trainData, trainTargets, batchsize); % arrange data in Batches
    [testData, testTargets] = makeBatches(testData, testTargets, batchsize); % arrange data in Batches
    Data = {trainData, testData};
    Targets = {trainTargets, testTargets};
    
%     clear dataStats testData testTargets trainData trainTargets
    %%%%% LEARNING %%%%%%
    [layerProp, err, trainTime] = BackPropagation(Data, layerProp, Targets, 10^-3, calcError, 1);
    
    parsave([rootpathResults 'BPdata' num2str(n) '.mat'], trainTime, layerProp, err)
end
