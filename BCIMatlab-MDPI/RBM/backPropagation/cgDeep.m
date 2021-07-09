function [f, df] = cgDeep(VV, reactFunct, Dim, Data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do backpropagation giving results of the differnces between actual and
% reconstructed data
%
% NEED TO SEE IF IT REALLY WORKS!!! Should be working and does not need a lot 
% 	more work to it
% 

L = Dim;
N = size(Data,1);
numLayers = size(L,1)-1;

% Do deconversion
 xxx = 0;
for jj = 1 : numLayers
    w{jj} = reshape( VV( xxx + 1 : xxx + (L(jj)+1) * L(jj+1) ), L(jj)+1,L(jj+1) );
    xxx = xxx + (L(jj)+1) * L(jj+1);
end

target = [Data ones(N,1)];
wtemp = [Data ones(N,1)];
for jj = 1 : numLayers
    targetout = reactFunct(jj).React(wtemp* w{jj}); 
    if jj ~= numLayers, wtemp = [targetout ones(N,1)]; wprobs{jj} = wtemp; end	
end    

% targetout = exp(wtemp*wClass);
% targetout = targetout./repmat(sum(targetout,2),1,size(target,2));
f = -1/N*sum(sum( target(:,1:end-1).*log(targetout) + (1-target(:,1:end-1)).*log(1-targetout)));
    
Ix = 1/N*(targetout-target(:,1:end));
dw = wtemp'*Ix; 
df = [dw(:)']'; 

wtemp2 = w{end};
for jj = 1 : numLayers-1
	Ix = (Ix*wtemp2').*wprobs{numLayers-jj}.*(1-wprobs{numLayers-jj});
	Ix = Ix(:,1:end-1); % current increment(taking out the bias)
	if jj ~= numLayers-1
		wtemp = wprobs{numLayers-jj-1}; 
	else
		wtemp = [Data ones(N,1)];
	end
	wtemp2 = w{numLayers-jj};
	dw =  wtemp'*Ix;
	df = [dw(:)' df(:)']';
end

end
