function varargout = rbmConstrastiveDUpdate(trainPorcentage, layerProp, Delta, nbt)

if nargin < 5, nbt = 100; end % number of batches per trail

[dw, dhb, dvb, dcw, dcb] = Delta{:};

% learning rate %
ew  = layerProp.epsilonw(trainPorcentage);
evb = layerProp.epsilonvb(trainPorcentage);
ehb = layerProp.epsilonhb(trainPorcentage);
if sum(strcmp(layerProp.trainingType, {'RBMclass'}))
    ecw = layerProp.epsiloncw(trainPorcentage);
    ecb = layerProp.epsiloncb(trainPorcentage);
end
% momentums %
mw = layerProp.momentumw(trainPorcentage);
mhb = layerProp.momentumhb(trainPorcentage);
mvb = layerProp.momentumvb(trainPorcentage);
if sum(strcmp(layerProp.trainingType, {'RBMclass'}))
    mcw = layerProp.momentumcw(trainPorcentage);
    mcb = layerProp.momentumcb(trainPorcentage);
end

% costs %
costw = layerProp.costw(trainPorcentage);
costvb = layerProp.costvb(trainPorcentage);
costhb = layerProp.costhb(trainPorcentage);
if sum(strcmp(layerProp.trainingType, {'RBMclass'}))
    costcw = layerProp.costcw(trainPorcentage);
    costcb = layerProp.costcb(trainPorcentage);
end

%%% UPDATES %%%
% increments
layerProp.weightsinc = mw*layerProp.weightsinc + ...
    ew*(dw/nbt - costw*layerProp.weights);
layerProp.visbiasinc = mvb*layerProp.visbiasinc + ...
    evb*(dvb/nbt - costvb*layerProp.visbiases);
layerProp.hidbiasinc = mhb*layerProp.hidbiasinc + ...
    ehb*(dhb/nbt - costhb*layerProp.hidbiases);
if sum(strcmp(layerProp.trainingType, {'RBMclass'}))
    layerProp.classWeightsinc = mcw*layerProp.classWeightsinc + ...
        ecw*(dcw/nbt - costcw*layerProp.classWeights);
    layerProp.classBiasesinc = mcb*layerProp.classBiasesinc + ...
        ecb*(dcb/nbt - costcb*layerProp.classBiases);
end

% Updates %
layerProp.weights   = layerProp.weights + layerProp.weightsinc;
layerProp.visbiases = layerProp.visbiases + layerProp.visbiasinc;
layerProp.hidbiases = layerProp.hidbiases + layerProp.hidbiasinc;
if sum(strcmp(layerProp.trainingType, {'RBMclass'}))
    layerProp.classWeights   = layerProp.classWeights + layerProp.classWeightsinc;
    layerProp.classBiases = layerProp.classBiases + layerProp.classBiasesinc;
end
varargout = {layerProp};
varargout = varargout(1:nargout);
end