function [varargout] = backProp(Data, Targets, layerProp, maxepoch, calcError, ECHO)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do back propagation taking use of the class 
% 
% 

if nargin < 3, error('Not enough input arguments'); end 

testData = [];
testTargets = [];
trainData = Data{1};
trainTargets = Targets{1};

if nargin < 4, maxepoch = 200; end
if nargin < 5, calcError = 0; end
if size(calcError,2) < 2, calcError(2) = 1;end
if nargin < 6, ECHO = 0; end 

if calcError(1) && size(Data,2) > 1 && size(Targets,2) > 1
    testData = Data{2};
    testTargets = Targets{2};
end

%%%% PREINITIALIZE WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%
numLayers = size(layerProp,2);
L = zeros(numLayers + 2 , 1 );
for i = 1 : numLayers
	L(i) = size(layerProp(i).weights,1);
    reactFunct(i) = layerProp(i).hidFunct;  
end
L(i+1) = size(layerProp(i).classWeights,1);
reactFunct(i+1) = layerProp(i).classFunct;
L(i+2) = layerProp(i).numClasses;
%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%

[nTrials, ~, nbt] = size(trainData);% [nTrials nInputs nbt] = size(trainData);
sizeErr      = floor(layerProp(end).trainEpochs/calcError(2));
testErr         = zeros(sizeErr, 1); 
testCrerr       = zeros(sizeErr, 1);
testNumClass    = zeros(sizeErr, size(trainTargets,2));
testRecErr      = zeros(sizeErr, 1);
trainErr        = zeros(sizeErr, 1); 
trainCrerr      = zeros(sizeErr, 1);
trainNumClass   = zeros(sizeErr, size(trainTargets,2));
trainRecErr     = zeros(sizeErr, 1);
ttError = 0;

sizeMB = nextpow2(nTrials);% size of a bigger of mini batch 

for epoch = 1 : maxepoch
    tt = 0;
        
    %%%%%%% COMPUTE TRAINING ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if calcError(1) && ~mod(epoch - 1, calcError(2)) ,
        ttError = ttError + 1;
        if ~isempty(trainData) && ~isempty(trainTargets) && ~strcmp(layerProp(1,i).trainingType,'RBMDeep')
            [trainErr(ttError), trainCrerr(ttError) trainNumClass(ttError,:)] = rbmClasificationError(trainData, trainTargets, layerProp);
        end
        trainRecErr(ttError) = rbmReconstructionError(trainData, layerProp, numLayers);
    end
    %%%%%%% COMPUTE TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if calcError(1) && ~mod(epoch - 1, calcError(2)),
        if ~isempty(testData) && ~isempty(testTargets) && ~strcmp(layerProp(1,i).trainingType,'RBMDeep')
            [testErr(ttError), testCrerr(ttError) testNumClass(ttError,:)] = rbmClasificationError(testData, testTargets, layerProp);
        end
        testRecErr(ttError) = rbmReconstructionError(testData, layerProp, numLayers);
    end
    
    for batch = 1:ceil(nbt/sizeMB)
        if ECHO, fprintf(1,'epoch %d batch %d\r',epoch,batch); end
        
        %%%%%%%%%%% COMBINE THE DOUBLE OF MINIBATCHES INTO 1 LARGER MINIBATCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tt = tt+1;
        if sizeMB*batch > nbt, sizeMB = nbt-sizeMB*(batch-1); end
        data = zeros(size(trainData,1)*sizeMB , size(trainData,2));
        targets = zeros(size(trainTargets,1)*sizeMB , size(trainTargets,2));
        for kk = 1:sizeMB
            data(size(trainData,1)*(kk-1)+1:size(trainData,1)*kk,:) = trainData(:,:,(tt-1)*sizeMB+kk);
            targets(size(trainTargets,1)*(kk-1)+1:size(trainTargets,1)*kk,:) = trainTargets(:,:,(tt-1)*sizeMB+kk);
        end
        
        %%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        max_iter=3; % Number of iterations %
        if epoch < 6  % First update top-level weights holding other weights fixed.
            N = size(data,1);
            wtemp = data;% ones(N,1)];
            for i = 1 : numLayers 
                wtemp = [wtemp ones(N,1)]; % add ones for the biases 
                w = [layerProp(i).weights; layerProp(i).hidbiases];
                wtemp = layerProp(i).hidFunct.React(wtemp * w,1);
            end         
            % output layer
            w = [layerProp(numLayers).classWeights; layerProp(numLayers).classBiases];
            VV = (w(:)')';
            Dim = [L(end-1); L(end)];
            [X, fX] = minimize(VV,'cgClassify',max_iter, reactFunct(end), Dim, wtemp,targets);
            w = reshape(X,L(end-1)+1,L(end));
            
            layerProp(numLayers).classWeights = w(1:L(numLayers+1),:);
            layerProp(numLayers).classBiases = w(L(numLayers+1)+1:end, :);
        else
            VV = [];
            for i = 1 : numLayers
                w = [layerProp(i).weights; layerProp(i).hidbiases];
                VV = [VV' w(:)']';
            end
            w = [layerProp(i).classWeights; layerProp(i).classBiases];
            VV = [VV' w(:)']';
            
            Dim = L;
            wtemp = data;% ones(N,1)];
            [X, fX] = minimize(VV,'cgClassify',max_iter, reactFunct, Dim, wtemp,targets);
            
            xxx = 0;
            for i = 1 : numLayers
                w = reshape( X( xxx + 1 : xxx + (L(i)+1) * L(i+1) ), L(i)+1,L(i+1) );
                xxx = xxx + (L(i)+1) * L(i+1);
                layerProp(i).weights = w(1:L(i),:);
                layerProp(i).hidbiases = w(L(i)+1:end, :);
                    
            end
            w = reshape( X( xxx + 1 : xxx + (L(i+1)+1) * L(i+2) ), L(i+1)+1,L(i+2) );
            xxx = xxx + (L(i+1)+1) * L(i+2);
            layerProp(numLayers).classWeights = w(1:L(i+1),:);
            layerProp(numLayers).classBiases = w(L(i+1)+1:end, :);
        end
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

end

Err = struct('type', ['Back Propagation Errors'], 'testErr', testErr, 'testCrerr', testCrerr, 'testNumClass', testNumClass,...
                                        'trainErr', trainErr, 'trainCrerr',trainCrerr, 'trainNumClass', trainNumClass);
varargout = {layerProp, Err};
varargout = varargout(1:nargout);

end
