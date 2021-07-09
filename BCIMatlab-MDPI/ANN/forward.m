function layerProp = forward(data, layerProp)

numLayers = size(layerProp,2);
numPatterns = size(data,1);

layerProp(1).activation = [data ones(numPatterns,1)]';

% Calculation of  Forward 
for i = 1:numLayers-1 % Calc of forward layer by layer
    layerProp(i).net = layerProp(i).activation'*[layerProp(i).weights' layerProp(i).bias']';
    if i < numLayers - 1
        % layerProp(i).net = layerProp(i).activation'*[layerProp(i).weights' layerProp(i).bias']';
        layerProp(i+1).activation = [logsig(layerProp(i).net(:,1:end-1)) ones(numPatterns,1)]'; %Calc exit of every layer
    else
        %  layerProp(i).activation = layerProp(i).activation(1:end-1,:);
        %  layerProp(i).net = layerProp(i).activation'*[layerProp(i).weights']';
        %  layerProp(i+1).activation = ( exp(layerProp(i).net)./repmat(sum(exp(layerProp(i).net),2),1,size(layerProp(i).net,2)) )';
        layerProp(i+1).activation = [logsig(layerProp(i).net)]';
    end
end

end