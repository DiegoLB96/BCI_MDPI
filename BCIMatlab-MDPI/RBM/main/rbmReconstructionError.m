function varargout = rbmReconstructionError(Data, layerProp, numLayers)
% This function goes from the feature vector through the different layers of the RBM
% and return reconstructing the feature vector.
% Note: This function does not take consideration of the labels of the
% label in the last layer if included

err_rc = 0;

[nTrials nInputs nbt] = size(Data);
for k = 1 : nbt
    data = Data(:,:,k);
    targetout = data;% targetout = [data ones(nTrials,1)];
    
    %%%%%%%%% FEATURE EXTRACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1 : numLayers
        IHidden = layerProp(j).hidFunct.Input(layerProp(j).weights,layerProp(j).hidbiases,targetout,0,0);
        RHidden = layerProp(j).hidFunct.React(IHidden);
        targetout = RHidden;
    end  
%     IClass = layerProp(end).classFunct.Input(layerProp(end).classWeights,layerProp(end).classBiases,targetout);
%     RClass = layerProp(end).classFunct.React(IClass);
 
    %%%%%%%%% RECONSTRUCTION DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = numLayers:-1:1
        IVisible = layerProp(j).visFunct.Input(layerProp(j).weights',layerProp(j).visbiases,targetout);% Input for visible units % 
		RVisible = layerProp(j).visFunct.React(IVisible); 
        targetout = RVisible;
    end

    err_rc = err_rc + 1/nTrials*sum(sum( (data(:,:) - targetout).^2 ));
end

err_rc = err_rc/nbt;
err_rc = err_rc/nInputs; % porcentage of error over the whole inputs

varargout = {err_rc};
varargout = varargout(1:nargout);
end