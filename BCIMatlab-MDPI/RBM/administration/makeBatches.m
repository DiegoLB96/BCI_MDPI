function varargout = makeBatches(data,targets,batchsize, reOrder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 02.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Put the data in a 3 dimensional matrix to simplify future matrix multiplications
% inside the RBM
%

if nargin < 4, reOrder = 1; end 
    totnum = size(data,1);
    fprintf(1, 'Size of the dataset= %5d \n', totnum);
    
    randomorder = 1:totnum;
    
    if reOrder
        s = RandStream.create('mrg32k3a');
        s.reset; %so we know the permutation of the training data
        randomorder = randperm(s,totnum);
%         rand('state',0); %so we know the permutation of the training data
%         randomorder = randperm(totnum);
    end
    
    nbt = floor(totnum/batchsize);
    if nbt == 0; nbt = 1; end
    [nI, nInputs]  =  size(data);
    nClasses = size(targets,2);
    if nbt ~= 1
        batchdata = zeros(batchsize, nInputs, nbt);
        batchtargets = zeros(batchsize, nClasses, nbt);
    else
        batchdata = zeros(nI, nInputs, nbt);
        batchtargets = zeros(nI, nClasses, nbt);
    end

    for b = 1:nbt
        if nbt ~= 1
            batchdata(:,:,b) = data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
            batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
        else
            batchdata(:,:,b) = data(randomorder, :);
            batchtargets(:,:,b) = targets(randomorder, :);
        end
    end;
	
	varargout = {batchdata, batchtargets, randomorder};
	varargout = varargout(1:nargout);
    
end
