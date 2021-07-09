function varargout = convertDataSFT(w, fs, label, FRange, TRange, windowSize, stepSize, dataType)
% converts to the frequency domain the data 
% and select the frequency bands
data = [];
targets = [];
for j = 1 : size(w,2)
    step = ceil( fs * stepSize );                 % Step for transformation
    window = ceil( fs * windowSize );               % Size of the window
    noverlap = window - step;             % Overlaping number of segments
    nfft = 2^nextpow2(window);
    for k = 1 : size(w,3)
        [S, fv, tv, P(:,:,k)] = spectrogram(w(:,j,k), window, noverlap, nfft, fs);
%         mesh(tv,fv,P(:,:,2))
    end
    
    P = 10*log10(abs( P ));
    idx = tv <= TRange(2) & tv >= TRange(1);
    % idx2 = fv <= FRange(1,2) & fv >= FRange(1,1); % idxtemp = Fv <= 18 & Fv>=13; idx = logical(idx - idxtemp);
    idx2 = zeros(size(fv,1),1);
    for n = 1 : size(FRange,1)
        idx2 = idx2 | fv <= FRange(n,2) & fv >= FRange(n,1);
    end
    Tv = tv(idx);
    Fv = fv(idx2);
    % mesh(Tv, Fv, P(idx2,idx,2))
    % set(gca,'yscale','log','ydir','reverse')
    Dtemp = P(idx2,idx,:);
        
    D = [];
    for m = 1 : size(Dtemp,3)
        D = [D Dtemp(:,:,m).'];
    end
    
    data = [data; D];
    
    targets = [targets; repmat(label(j), size(D,1) , 1 )];
    
end

varargout = {data, targets, Tv, Fv};
varargout = varargout(1:nargout);
end