function varargout = oneRep(classes, numClasses)
% if nargin < 2, numClasses = max(classes); end
e = eye(numClasses);
e = fliplr(e);

classes = e(classes,:);

varargout = {classes};
varargout = varargout(1:nargout);
end