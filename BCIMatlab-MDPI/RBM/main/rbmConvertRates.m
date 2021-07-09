function varargout = rbmConvertRates(LR, CT, MOM, typeLayer, TV, dim)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 20.03.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the learning rates or convert the learning rates, costs and 
% momentums into functions
%
% SYNTAX
% [LR, CT, MOM] = rbmConvertRates(LR, CT, MOM, typeLayer, TV, dim)
% 
% DEFINITION
% Initialize the learning rates or convert the learning rates, costs and 
% momentums into functions
% 
% %% Inputs %%
% LR: entry of the learning rates 
% CT: entry of the costs
% MOM: entry of the momentums
%       Note: they can be entered in a cell array as a function or as numeric  
% typeLayer: if empty it creates depending of the type of layer specified
% TV: adaps the rates also depending on the type of variable layer(Not implemented)
% dim: number of dimensions to create the rates
%
% %% Outputs %%
% LR: learning rate in a function representation
% CT: Costs in a function representation
% MOM: Momentums in a function representation

if ~iscell(typeLayer), typeLayer = {typeLayer}; end
if nargin < 4, typeLayer = 'Binary'; end
if nargin < 5, TV = 'Binary'; end
if nargin < 6, dim = 3; end


numLayers = size(typeLayer,2);
for j = 1: numLayers
    for k = 1 : dim
       try 
          if isnumeric(LR{j,k}), LR(j,k) = {eval(['@(x)' num2str(LR{j,k})])}; elseif isempty(LR{j,k}), error(1); end, 
      catch, 
        if strcmp(typeLayer{1,j}, 'Gauss')
            LR(j,k) = {@(x)0.005*(1-0.75*x^.5)}; % LEARNING RATE
        elseif strcmp(typeLayer{1,j}, 'Softmax') 
            LR(j,k) = {@(x)0.005*(1-0.75*x^.5)};
        elseif strcmp(typeLayer{1,j}, 'Rlin')
            LR(j,k) = {@(x)0.01*(1-0.75*x^.5)};%*(1-x^.5)
        else
            LR(j,k) = {@(x)0.01*(1-0.75*x^.5)};
        end
      end
      try 
          if isnumeric(CT{j,k}), CT(j,k) = {eval(['@(x)' num2str(CT{j,k})])}; elseif isempty(CT{j,k}), error(1); end, 
      catch, 
          CT(j,k) = {@(x)0.0002}; 
      end
      try 
          if isnumeric(MOM{j,k}), MOM(j,k) = {eval(['@(x)' num2str(MOM{j,k})])}; elseif isempty(MOM{j,k}), error(1); end, 
      catch, 
          if strcmp(typeLayer{1,j}, 'Gauss') 
              MOM(j,k) = {@(x)0.5*(1+floor(x*2.5)/5)}; % MOM(j,k) = {@(x)(x<=20)*0.5 + (x>20)*0.9};
          elseif strcmp(typeLayer{1,j}, 'Softmax') 
              MOM(j,k) = {@(x)0.5*(1+0.25*x^.9)};
          elseif strcmp(typeLayer{1,j}, 'Rlin') 
              MOM(j,k) = {@(x)0.5*(1+0.25*x^.9)};
          else
              MOM(j,k) = {@(x)0.5*(1+0.75*x^.5)};
          end
          
      end 
   end
   TV = typeLayer{1,j};
end

varargout = {LR, CT, MOM};
varargout = varargout(1:nargout);

      
end