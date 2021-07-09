function [varargout] = backPropDeep(Data, Targets, layerProp, maxepoch, calcError, calcErrorMod, ECHO)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do back propagation of RBM as a deep auto encoder
%
% [Weights, Err] = backPropDeep(Data, Targets, layerProp, maxepoch, calcError, calcErrorMod, ECHO)
%
%
%
%
% Note: I doesn't return a layerProp like the rest since it becomes something 
%  	different having a double set of number of layers  
% Note2: The output 'Weights' contains also the bias for now each of the invidual 
% 	layers
%
% TODO: it is not compleatly implemented it requieres the use and updates of the 
% 	weights in a better manner so it can remmain a RBM

if nargin < 3, error('Not enough input arguments'); end 

testData = [];
testTargets = [];
trainData = Data{1};
trainTargets = Targets{1};

if nargin < 4, maxepoch = 200; end
if nargin < 5, calcError = 0; end
if nargin < 6, calcErrorMod = 1; end
if nargin < 7, ECHO = 0; end 

if calcError && size(Data,2) > 1 && size(Targets,2) > 1
    testData = Data{2};
    testTargets = Targets{2};
end

%%%% PREINITIALIZE WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%
numLayers = size(layerProp,2);
L = zeros(numLayers*2 + 1 , 1 );

newNumLayers = 0;
	%%% CONSTRUCTION MODEL DATA %%%
for j = 1 : numLayers
    newNumLayers = newNumLayers + 1;
	Weights(newNumLayers,:,:) = [layerProp(j).weights; layerProp(j).hidbiases];
	L(newNumLayers) = size(layerProp(j).weights,1);
    reactFunct(newNumLayers) = layerProp(j).hidFunct;  
end

	%%% RECONSTRUCTION DATA %%%
for j = numLayers : -1 :1
    newNumLayers = newNumLayers + 1;
	Weights(newNumLayers,:,:) = [layerProp(j).weights'; layerProp(j).visbiases];
    L(newNumLayers) = size(layerProp(j).weights,2);
    reactFunct(newNumLayers) = layerProp(j).visFunct;
end
L(newNumLayers+1) = L(1);

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS %%%%%%%%%%%%%%%%%%%%%%%%%%%

[nTrials, ~, nbt] = size(trainData);% [nTrials nInputs nbt] = size(trainData);

sizeErr         = maxepoch/calcErrorMod;
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
    if calcError && ~mod(epoch - 1, calcErrorMod),
        ttError = ttError + 1;
        trainRecErr(ttError) = rbmReconstructionError(trainData, layerProp, numLayers);
    end
    %%%%%%% COMPUTE TEST ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%
    if calcError && ~mod(epoch - 1, calcErrorMod),
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
        VV = [];
          
        for j = 1 : newNumLayers
			wT = Weights(j,:,:);
            VV = [VV' wT(:)']';
        end
        
        Dim = L;
        wtemp = [data];% ones(N,1)];
        [X, fX] = minimize(VV,'cgClassifyDeep',max_iter, reactFunct, Dim, wtemp);
        
        xxx = 0;
        for j = 1 : newNumLayers
            wT = reshape( X( xxx + 1 : xxx + (L(j)+1) * L(j+1) ), L(j)+1,L(j+1) );
            xxx = xxx + (L(j)+1) * L(j+1);
			Weights(newNumLayers,:,:) = w;
            
%            layerProp(i).weights = w(1:L(i),:);
%            layerProp(i).hidbiases = w(L(i)+1:end, :);
        end
        %%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH 3 LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end

end

Err = struct('type', ['Back Propagation Errors'], 'testErr', testErr, 'testCrerr', testCrerr, 'testNumClass', testNumClass,...
                                        'trainErr', trainErr, 'trainCrerr',trainCrerr, 'trainNumClass', trainNumClass);
varargout = {Weights, reactFunct, Err};
varargout = varargout(1:nargout);

end
