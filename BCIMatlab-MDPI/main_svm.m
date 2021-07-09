% Supported Vector Machines
% David Balderas
% TEC Monterrey
% Training Data load
addpath(['Extras']);
subject = 1;                 % Subject Number
dataTransform = 'SFT'; % SFT, Wavelets, Time
[trainData, trainTargets, testData, testTargets, dataStats] = loadData();
trainEpochs = 50; % number of epochs to train each layer of the RBM 
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
    
    %     options = optimset('Display', 'iter', 'TolX', 1e-4);
        % svmStruct = svmtrain(trainData,tTargets,'kernel_function','rbf');
    %     svmStruct = svmtrain(trainData(1:round(size(trainData,1)/20),:), ...
    %         tTargets(1:round(size(trainData,1)/20),:),...
    %         'kernel_function','rbf',...
    %         'BoxConstraint',2e-2,...
    %         'options', options);
    
    tic
    svmModel = fitcsvm(trainData(1:round(size(trainData,1)/1),:), ...
        tTargets(1:round(size(trainData,1)/1),:),...
        'KernelFunction','rbf',...
        'BoxConstraint',1e-1);
    trainTime = toc;
    
    [Group, score] = predict(svmModel, trainData);
    tTargets = ones(size(trainTargets,1),1);
    tTargets(trainTargets(:,2)==1) = -1;
    svmTrainError = sum(tTargets ~= Group)/size(trainTargets,1);
    
    tic
    [Group, score] = predict(svmModel, testData);
    testTime = toc;
    tTargets = ones(size(testTargets,1),1);
    tTargets(testTargets(:,2)==1) = -1;
    svmTestError = sum(tTargets ~= Group)/size(testTargets,1);
    
    % toc
    parsave([rootpathResults 'SVMdata' num2str(n) '.mat'], trainTime, testTime, svmModel,svmTrainError,svmTestError)
end
