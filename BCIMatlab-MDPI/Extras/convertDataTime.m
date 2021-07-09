function varargout = convertDataTime(w, fs, label, FRange, TRange, windowSize, stepSize, dataType)
% converts and filters the data in the time domain
data = [];
targets = [];
for j = 1 : size(w,2)
    step = ceil( fs * stepSize );                 % Step for transformation
    window = ceil( fs * windowSize );               % Size of the window
    noverlap = window - step;             % Overlaping number of segments
    numSeg = (TRange(2) - TRange(1))*fs - window + 1; %*fs - window + 1;
    numSeg = floor(numSeg / step) + 1;
    
    N = size(w,1);
    T = N/fs;
    [B,A] = butter(5,FRange/(fs/2));
    tv = 0:1/fs:T-1/fs; tv = tv';
    P = filtfilt(B,A,w(:,j,:));
    P = squeeze(P);
    idx = tv <= TRange(2) & tv >= TRange(1);
    Tv = tv(idx);
    
    Dt = P(idx,:);
    
    for n = 1 : numSeg  %step : size(Dtemp,1)
        Dtemp(:, n, :) = Dt((n-1)*step + 1 : (n-1)*step + window, :);
    end

    D = [];
    for m = 1 : size(Dtemp,3)
        D = [D Dtemp(:,:,m).'];
    end
    
    data = [data; D];
    
    targets = [targets; repmat(label(j), size(D,1) , 1 )];
    
    Fv = zeros(fs,1);
    % for congruency with other methods in this case it does not return the
    % frequency vector, but the number of segments the interval is split
    % into
end

varargout = {data, targets, Tv, Fv};
varargout = varargout(1:nargout);
end