% LDA
% David Balderas
% TEC Monterrey
% Training Data load
addpath(['Extras']);
subject = 1;                 % Subject Number
dataTransform = 'SFT'; % SFT, Wavelets, Time
[trainData, trainTargets, testData, testTargets, dataStats] = loadData();
trainEpochs = 20; % number of epochs to train each layer of the RBM 
LR = 0.05; % Learning rate eta
% Tau = [1000,1000];
Mom = 0.05; % distance of radios of update
calcError = [1];

parfor n = stSubj:numSubj
    g = load([rootpathResults 'Data' num2str(n) '.mat'])
    trainData = g.trainData;
    trainTargets= g.trainTargets;
    testData = g.testData;
    testTargets = g.testTargets;
    dataStats = g.dataStats;
    numFeatures     = size(trainData, 2); % equal to the number of weights
    numClasses      = size(trainTargets, 2);

    tTargets = ones(size(trainTargets,1),1);
    tTargets(trainTargets(:,2)==1) = -1;
    
%%%% TRAIN LDA %%%%
    tic
    ldaModel = fitcdiscr(trainData(1:round(size(trainData,1)/1),:), ...
        tTargets(1:round(size(trainData,1)/1),:));
    trainTime = toc;
    
    [Group, score] = predict(ldaModel, trainData);
    tTargets = ones(size(trainTargets,1),1);
    tTargets(trainTargets(:,2)==1) = -1;
    ldaTrainError = sum(tTargets ~= Group)/size(trainTargets,1);
    
    tic
    [Group, score] = predict(ldaModel, testData);
    testTime = toc;
    tTargets = ones(size(testTargets,1),1);
    tTargets(testTargets(:,2)==1) = -1;
    ldaTestError = sum(tTargets ~= Group)/size(testTargets,1);
    

    parsave([rootpathResults 'LDAdata' num2str(n) '.mat'], trainTime, testTime, ldaModel,ldaTrainError,ldaTestError)
end
