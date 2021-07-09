function [varargout] = rbmDiscriminativeUpdate(trainPorcentage, layerProp, classLayerProp, Delta, nbt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   03.07.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBM Discriminative Update
%
% This function for the moment doesnt do anything for the future
% it could be erased


if nargin < 5, nbt = 100; end % number of batches per trail

[dw, dhb, dcw] = Delta{:};

% learning rate %
ew  = layerProp.epsilonw(trainPorcentage);
ehb = layerProp.epsilonhb(trainPorcentage);
ecw = classLayerProp.epsilonw(trainPorcentage);

% momentums %
mw  = layerProp.momentumw(trainPorcentage);
mhb = layerProp.momentumhb(trainPorcentage);
mcw = classLayerProp.momentumw(trainPorcentage);

% costs %
costw  = layerProp.costw(trainPorcentage);
costhb = layerProp.costhb(trainPorcentage);
costcw = classLayerProp.costw(trainPorcentage);

%%% UPDATES %%%
% increments
layerProp.weightsinc = mw*layerProp.weightsinc + ...
    ew*(dw/nbt - costw*layerProp.weights);
layerProp.hidbiasinc = mhb*layerProp.hidbiasinc + ...
    ehb*(dhb/nbt - costhb*layerProp.hidbiases);
classLayerProp.weightsinc = mcw*classLayerProp.weightsinc + ...
     ecw*(dcw/nbt - costcw*classLayerProp.weights);


% Updates %
layerProp.weights       = layerProp.weights + layerProp.weightsinc;
layerProp.hidbiases     = layerProp.hidbiases + layerProp.hidbiasinc;
classLayerProp.weights  = classLayerProp.weights + classLayerProp.weightsinc;

varargout = {layerProp,classLayerProp};
varargout = varargout(1:nargout);
end
