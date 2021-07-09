



if ~isempty(strfind(rootpathRead,'BCICompetition'))
    T = 8;                      % [s] Time of experiment
    
    j = 0;
    label = [];
    w = [];
    FRange = [8 30];            % [Hz] Frequency range
    TRange = [2 8];			% [s] Time range
    windowSize = 1.0;			% [s] for SFFT
    stepSize = 0.01;			% [s] for SFFT
    downSampFac = 4;            % for Time form
    chosenLabels = [1,2];       % left and right hand
    I = filesep;
    for m = 1 : 9
        for n = 1 : 3
            file2read = [rootpathRead 'BCICIV_2b_gdf' I 'B0' num2str(m*100+n) 'T.gdf'];
            [s, h] = sload(file2read);
            
            fs = h.SampleRate;          % [Hz] sample frequency
            triger = h.TRIG;
            dt = 1/fs;                  % [s] sample time
            N =  T*fs;                  % [#] Number of samples taken of the output
            for i = 1 : size(triger,1)
                
                if ~isfield(h, 'ArtifactSelection') || ~h.ArtifactSelection(i)
                    j = j + 1;
                    w(:,j,1) = s(triger(i):triger(i) + N - 1, 2);      % C3
                    w(:,j,2) = s(triger(i):triger(i) + N - 1, 1);      % Cz
                    w(:,j,3) = s(triger(i):triger(i) + N - 1, 3);      % C4
                    label(j) = h.Classlabel(i);
                end
            end
        end
    end
    
    if strcmp(dataTransform,'Time')
        w = w(1:downSampFac:end,:,:); fs = fs/downSampFac;
    end
    
    [data targets,Tv, Fv] = convertData(w, fs, label, FRange, TRange, windowSize, stepSize, dataTransform);
    [trainData trainTargets] = oneRepClasses(data, targets, chosenLabels);
    
    j = 0;
    label = [];
    w = [];
    
    
    for m = 1 : 9
        for n = 4 : 5
            file2read = [rootpathRead 'BCICIV_2b_gdf' I 'B0' num2str(m*100+n) 'E.gdf'];
            [s, h] = sload(file2read);
            classLabel = load([rootpathRead  'true_labels_set2b' I 'B0' num2str(m*100+n) 'E.mat']);
            fs = h.SampleRate;          % [Hz] sample frequency
            triger = h.TRIG;
            dt = 1/fs;                  % [s] sample time
            N =  T*fs;                  % [#] Number of samples taken of the output
            for i = 1 : size(triger,1)
                
                if ~isfield(h, 'ArtifactSelection') || ~h.ArtifactSelection(i)
                    j = j + 1;
                    w(:,j,1) = s(triger(i):triger(i) + N - 1, 2);      % C3
                    w(:,j,2) = s(triger(i):triger(i) + N - 1, 1);      % Cz
                    w(:,j,3) = s(triger(i):triger(i) + N - 1, 3);      % C4
                    label(j) = h.Classlabel(i);
                end
            end
        end
    end
    
    if strcmp(dataTransform,'Time')
        w = w(1:downSampFac:end,:,:); fs = fs/downSampFac;
    end
    
    if isempty(w)
        testData = [];
        testTargets = [];
    else
        [data, targets,Tv, Fv] = convertData(w, fs, label, FRange, TRange, windowSize, stepSize, dataTransform);
        [testData testTargets] = oneRepClasses(data, targets, chosenLabels);
    end
    %% NORMALIZE DATA
    totMean = mean(mean(trainData,2));%-max(abs(mean(trainData,2))); %
    %     totStd = 2*sqrt(mean(var(trainData,0,2)));% max(abs(std(trainData,0,2))); %
    totStd = 2*(mean(std(trainData,0,2)));
    
    trainData = trainData-totMean; % sign(trainData).*(abs(trainData) - alpha);
    testData = testData-totMean; % sign(testData).*(abs(testData) - alpha);
    
    trainData =  trainData./totStd;
    testData = testData./totStd;
    
    %%
    dataStats = struct('T',T,'fs',fs, 'totMean',totMean, 'totStd',totStd,...
        'Fv',Fv,'Tv',Tv,...
        'FRange',FRange,'TRange',TRange,...
        'Type',['WFL EMG Data'],...
        'Coments',[dataTransform, '\nWindow size:' num2str(windowSize) ' Step size:' num2str(stepSize),...
        '\nFrequency Range:' num2str(FRange) '-' num2str(FRange), '\nTime Range:' num2str(TRange) '-' num2str(TRange) ,...
        '\Normallized data ']);
end