function varargout = oneRepClasses(data, targets, chosenLabels, numClasses)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   03.07.2011-last modified 03.08.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBM DEFINITIONS
%
% SYNTAX
% [data, targets] = oneRepClasses(data, targets, chosenLabels, numClasses)
%  
% DEFINITION
% Creats a one rRepresentation of the classes (e.g. class1 = [0,1] and 
% class2 = [1,0]) and removes the vectors from data and targets that are 
% not used for training  
%
% Abbreviations:
%%% INPUTS %%%
% data: An array with the data to train the RBM
% targets:
% chosenLabels:chosenLabels
% numClasses:
% 
%%% OUTPUTS %%% 
% data: organized data array incluing only the vectors with the defined
%       classes
% targets:organized targets array incluing only the vectors with the 
%       defined classes, and being in a one representation (e.g. class1 =
%       [0,1] and class2 = [1,0]). 
% 
% Note that targets are not a binary representation

if nargin < 3 || isempty(chosenLabels), chosenLabels = 1:max(targets); end
if nargin < 4, numClasses = max(targets); end

idx = zeros(size(targets,1),1);
for m = 1 : size(chosenLabels,2)
    idx = idx + (targets == chosenLabels(m));
end

idx = idx >= 1;
if size(targets,1)~=1,
    data = data(idx, :);
    targets = targets(idx,:); % remove unnecesary classes
end
targets = oneRep(targets, numClasses);

idx2 = zeros(1,size(targets,2));
idx2(chosenLabels) = 1;
idx2 = fliplr(idx2); idx2 = logical(idx2);
targets = targets(:, idx2);

varargout = {data, targets};
varargout = varargout(1:nargout);
end
