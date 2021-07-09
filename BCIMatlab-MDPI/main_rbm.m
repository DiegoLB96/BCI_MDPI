%% NEURAL NETWORKS
% VARIABLES FOR NEURAL NETWORKS
%%% RBM toolbox %%%
I = filesep;
addpath([toolboxPath I 'RBM'],...
    [toolboxPath I 'RBM' I 'main'],...
    [toolboxPath I 'RBM' I 'backPropagation'],...
    [toolboxPath I 'RBM' I 'administration']);
addpath(['Extras']);
subject = 1;                 % Subject Number
dataTransform = 'SFT'; % SFT, Wavelets, Time
[trainData, trainTargets, testData, testTargets, dataStats] = loadData();

trainEpochs = 50; % number of epochs to train each layer of the RBM 
ngibbs = 1; % number of Gibbs samples for the RBM 
typeVisible = 'Gauss'; % Form of the input data to be use as visible or input layer
typeLayer = {'Binary'};% Defined hidden layers 
typeClass = 'Softmax';
trainingType = 'RBMclass';  % RBMclass: RBM Using Classifier, 
                            
neuronPerLayer = [];
calcError = [1,4]; % [1] to calculate the error, [0] Default
LR = {}; %0.001,.001,.001;0.01,0.01,0.01}; % Learning rate
CT = {}; % Cost
MOM = {}; % Momentums

parfor n = stSubj:numSubj
    g = load([rootpathResults 'Data' num2str(n) '.mat']);
            
    trainData = g.trainData;
    trainTargets= g.trainTargets;
    testData = g.testData;
    testTargets = g.testTargets;
    dataStats = g.dataStats;
    g = [];
    numFeatures     = size(trainData, 2);
    numClasses      = size(trainTargets, 2);
    
    %%%%%%%%%% RESTRICTED BOLTZMANN MACHINES %%%%%%%%%%%%
    %%%%% INITIALIZE LAYERS %%%%%
    layerProp = rbmInitLayers(numFeatures, trainingType, neuronPerLayer, typeVisible, typeLayer, trainEpochs, ...
            LR, CT, MOM, ngibbs,...
            numClasses, typeClass, {}, {}, {}, dataStats.totStd, dataStats.totMean);
    
    %%%% Arrange data in Batches %%%
    batchsize = 2^(floor((nextpow2(size(trainData,1))-2)/2));
    [trainData, trainTargets] = makeBatches(trainData, trainTargets, batchsize); % arrange data in Batches
    [testData, testTargets] = makeBatches(testData, testTargets, batchsize); % arrange data in Batches
    Data = {trainData, testData};
    Targets = {trainTargets, testTargets};
    
    %%%%% LEARNING %%%%%%
    [layerProp, ErrRBM, ErrBP, trainTime, trainTimeBp] = RBM(Data, layerProp, Targets, [], calcError, 0);%, ECHO);
    
    parsave([rootpathResults 'RBMdata' num2str(n) '.mat'], trainTime, trainTimeBp, layerProp, ErrRBM, ErrBP)
end


