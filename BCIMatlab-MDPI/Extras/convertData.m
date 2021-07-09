function [varargout] = convertData(w, fs, label, FRange, TRange, windowSize, stepSize, dataType)
% administration file to convert the data into the different formats

switch dataType
    case 'Time'
        [data, targets, Tv, Fv] = convertDataTime(w, fs, label, FRange, TRange, windowSize, stepSize, dataType);
    case 'SFT'    %%%% Windowed Fourier Transform(Short-Time Fourier Transform) %%%%%
        [data, targets, Tv, Fv] = convertDataSFT(w, fs, label, FRange, TRange, windowSize, stepSize, dataType);
    case 'Wavelets'
        [data, targets, Tv, Fv] = convertDataWavelets(w, fs, label, FRange, TRange, windowSize, stepSize, dataType);
    case 'Wave2'
        
end


totnum = size(data,1);
% fprintf(1, 'Size of the dataset= %5d \n', totnum);

%%
varargout = {data, targets, Tv, Fv};
varargout = varargout(1:nargout);

end
