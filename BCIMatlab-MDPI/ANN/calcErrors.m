function [varargout] = calcErrors(Data, Targets, layerProp)
[nTrials, ~, nbt] = size(Data);
nOutputs = size(Targets,2);
rErr = 0;
cErr = 0;
regErr = 0;
numLayers = size(layerProp,2);

for k = 1 : nbt
    data = Data(:,:,k);
    tarjets = Targets(:,:,k)';
    
    layerProp = forward(data, layerProp);
    
    [~,J] = max(layerProp(numLayers).activation,[],1);
    [~,J1] = max(tarjets,[],1);
    
    cErr = cErr + size(find(J == J1),2);
    
    rErr = sum((tarjets - layerProp(numLayers).activation).^2);
    
    regErr = regErr + sum(rErr);%/nTrials;
end
regErr = regErr./(nTrials*nOutputs*nbt);
classErr = cErr./(nTrials*nbt);

varargout = {classErr, regErr};
varargout = varargout(1:nargout);
end