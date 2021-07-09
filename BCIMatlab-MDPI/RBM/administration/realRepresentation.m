function varargout = realRepresentation(targets, chosenLabels)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creats a real value representation of the classes
% It is used for rbmDiscriminative
% 
if nargin < 2, chosenLabels = 0; end

% for i = 1 : size(targets)
   [~, Targets] = max(targets,[],2); 
% end

varargout = {Targets};
varargout = varargout(1:nargout);
end
