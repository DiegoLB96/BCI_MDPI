%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOAD BCI Competition Dataset %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trainData, trainTargets, testData, testTargets, dataStats] = loadData(subject)
T = 8;                      % [s] Time of experiment
j = 0;
label = [];
w = [];
U = [];
FRange = [8 13;22 30];            % [Hz] Frequency range
TRange = [4 7];			% [s] Time range
windowSize = 1.0;			% [s] for SFFT
stepSize = 0.01;			% [s] for SFFT
downSampFac = 4;            % for Time form
chosenLabels = [1,2];       % left and right hand
I = filesep;
si = [];
for n = 1 : 3
    file2read = ['BCICIV_2b_gdf' I 'B0' num2str(str2double(subject)*100+n) 'T.gdf'];
    [s, h] = sload(file2read);
    disp(h)
    %         s = s(:,1:3);
    fs = h.SampleRate;          % [Hz] sample frequency
    triger = h.TRIG;
    dt = 1/fs;                  % [s] sample time
    N =  T*fs;                  % [#] Number of samples taken of the output
    
    btemp = find(h.EVENT.TYP == 276 | h.EVENT.TYP == 277);
    btemp = [btemp; find(h.EVENT.TYP >= 1077 & h.EVENT.TYP <= 1081)];
    R = struct();
    if ~isempty(btemp)
        Umean = [];
        Ymean = [];
        for m = 1 : size(btemp)
            pos = h.EVENT.POS(btemp(m));
            dur = h.EVENT.DUR(btemp(m));
            Ymean = [Ymean; s(pos:pos+dur,1:3)];
            U1 = s(pos:pos+dur,4)-s(pos:pos+dur,6);
            U2 = s(pos:pos+dur,5)-s(pos:pos+dur,6);
            Umean = [Umean; [U1 U2]];
        end
        [R,S] = regress_eog([Ymean Umean],[1:3],[4:5]);
        % R.r1 is the sparse matrix that gives the corrected signal, it
        % has a extra column that corrects the offset of the signal. It
        % uses the covariance matrix between the signals and the noise.
        % It also removes the baseline of the EOG with the extra
        % columns of ones
        %             B = -(covm(Umean,Umean)\covm(Umean,Ymean));
    end
    if ~isfield(R,'r1')
        R = prevR;%zeros(size(prevB));
    end
    prevR = R;
    %         B = zeros(size(prevB));
    
    for i = 1 : size(triger,1)
        
        if ~isfield(h, 'ArtifactSelection') || ~h.ArtifactSelection(i)
            
            Y = s(triger(i):triger(i) + N - 1,1:3);
            U(:,1) = s(triger(i):triger(i) + N - 1,4)-s(triger(i):triger(i) + N - 1,6);
            U(:,2) = s(triger(i):triger(i) + N - 1,5)-s(triger(i):triger(i) + N - 1,6);
            
            % S = Y-U*B;
            % the extra columns of ones is to remove the baseline of the EOG
            S =  [ones(size(Y,1),1) Y U]*R.r1;
            S = S(:,1:3);
            j = j + 1;
            w(:,j,1) = S(:, 2);      % C3
            w(:,j,2) = S(:, 1);      % Cz
            w(:,j,3) = S(:, 3);      % C4
            label(j) = h.Classlabel(i);
            
        end
    end
    
    wtemp = find(sum(sum(isnan(w),1),3)~=0);% element with NaN
    w(:,wtemp,:) = [];
    wtemp = find(sum(sum(isinf(w),1),3));% element with inf
    w(:,wtemp,:) = [];
    wtemp = sum(sum(w == 0, 1) == N, 3)>0;% electrode filled with zeros
    w(:,wtemp,:) = [];
    
    si = [si, size(w,2)];
end

if strcmp(dataTransform,'Time')
    w = w(1:downSampFac:end,:,:); fs = fs/downSampFac;
end

[data targets,Tv, Fv] = convertData(w, fs, label, FRange, TRange, windowSize, stepSize, dataTransform);
[trainData trainTargets] = oneRepClasses(data, targets, chosenLabels);

j = 0;
label = [];
w = [];

for n = 4 : 5
    file2read = [rootpathRead 'BCICIV_2b_gdf' I 'B0' num2str(str2double(subject)*100+n) 'E.gdf'];
    [s, h] = sload(file2read);
    classLabel = load([rootpathRead  'true_labels_set2b' I 'B0' num2str(str2double(subject)*100+n) 'E.mat']);
    classLabel = classLabel.classlabel;
    fs = h.SampleRate;          % [Hz] sample frequency
    triger = h.TRIG;
    dt = 1/fs;                  % [s] sample time
    N =  T*fs;                  % [#] Number of samples taken of the output
    
    btemp = find(h.EVENT.TYP == 276 | h.EVENT.TYP == 277);
    btemp = [btemp; find(h.EVENT.TYP >= 1077 & h.EVENT.TYP <= 1081)];
    R = struct();
    if ~isempty(btemp)
        Umean = [];
        Ymean = [];
        for m = 1 : size(btemp)
            pos = h.EVENT.POS(btemp(m));
            dur = h.EVENT.DUR(btemp(m));
            Ymean = [Ymean; s(pos:pos+dur,1:3)];
            U1 = s(pos:pos+dur,4)-s(pos:pos+dur,6);
            U2 = s(pos:pos+dur,5)-s(pos:pos+dur,6);
            Umean = [Umean; [U1 U2]];
        end
        [R,S] = regress_eog([Ymean Umean],[1:3],[4:5]);
        % R.r1 is the sparse matrix that gives the corrected signal, it
        % has a extra column that corrects the offset of the signal. It
        % uses the covariance matrix between the signals and the noise.
        % It also removes the baseline of the EOG with the extra
        % columns of ones
        %             B = -(covm(Umean,Umean)\covm(Umean,Ymean));
    end
    if ~isfield(R,'r1')
        R = prevR;%zeros(size(prevB));
    end
    prevR = R;
    %         B = zeros(size(prevB));
    
    for i = 1 : size(triger,1)
        
        if ~isfield(h, 'ArtifactSelection') || ~h.ArtifactSelection(i)
            
            Y = s(triger(i):triger(i) + N - 1,1:3);
            U(:,1) = s(triger(i):triger(i) + N - 1,4)-s(triger(i):triger(i) + N - 1,6);
            U(:,2) = s(triger(i):triger(i) + N - 1,5)-s(triger(i):triger(i) + N - 1,6);
            
            % S = Y-U*B;
            % the extra columns of ones is to remove the baseline of the EOG
            S =  [ones(size(Y,1),1) Y U]*R.r1;
            S = S(:,1:3);
            j = j + 1;
            w(:,j,1) = S(:, 2);      % C3
            w(:,j,2) = S(:, 1);      % Cz
            w(:,j,3) = S(:, 3);      % C4
            label(j) = classLabel(i);
        end
    end
    
    wtemp = find(sum(sum(isnan(w),1),3)~=0);% element with NaN
    w(:,wtemp,:) = [];
    wtemp = find(sum(sum(isinf(w),1),3));% element with inf
    w(:,wtemp,:) = [];
    wtemp = sum(sum(w == 0, 1) == N, 3)>0;% electrode filled with zeros
    w(:,wtemp,:) = [];
    
    si = [si, size(w,2)];
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

trainData = trainData-totMean; % sign(trainData).*(abs(trainData) - alpha);
testData = testData-totMean; % sign(testData).*(abs(testData) - alpha);

totStd = (max(max(abs(trainData))));
totStd = 2*(mean(std(trainData,0,2)));

trainData =  trainData./totStd;
testData = testData./totStd;
% trainData = reshape(trainData, [size(trainData,1) size(Fv,1) 1 3]);
% testData = reshape(testData, [size(testData,1) size(Fv,1) 1 3]);

%%
dataStats = struct('T',T,'fs',fs, 'totMean',totMean, 'totStd',totStd,...
    'Fv',Fv,'Tv',Tv,...
    'FRange',FRange,'TRange',TRange,...
    'Type','BCI COmpetition',...
    'Coments',[dataTransform, '\n Window size:' num2str(windowSize) ' Step size:' num2str(stepSize),...
    '\Normallized data ']);

% % Run 1
% trainData = [trainData; testData];
% testData = trainData(1 : si(1)*250, :);
% trainData(1 : si(1)*250, :) = [];
% trainTargets = [trainTargets; testTargets];
% testTargets = trainTargets(1 : si(1)*250, :);
% trainTargets (1 : si(1)*250, :) = [];

% % Run 2
% trainData = [trainData; testData];
% testData = trainData(si(1)*250 + 1 : si(2)*250, :);
% trainData(si(1)*250 + 1 : si(2)*250, :) = [];
% trainTargets = [trainTargets; testTargets];
% testTargets = trainTargets(si(1)*250 + 1 : si(2)*250, :);
% trainTargets (si(1)*250 + 1 : si(2)*250, :) = [];

% % Run 3
% trainData = [trainData; testData];
% testData = trainData(si(2)*250 + 1 : si(3)*250, :);
% trainData(si(2)*250 + 1 : si(3)*250, :) = [];
% trainTargets = [trainTargets; testTargets];
% testTargets = trainTargets(si(2)*250 + 1 : si(3)*250, :);
% trainTargets (si(2)*250 + 1 : si(3)*250, :) = [];

% % Run 4
% trainData = [trainData; testData];
% testData = trainData(si(3)*250 + 1 : si(3)*250 + si(4)*250, :);
% trainData(si(3)*250 + 1 : si(3)*250 + si(4)*250, :) = [];
% trainTargets = [trainTargets; testTargets];
% testTargets = trainTargets(si(3)*250 + 1 : si(3)*250 + si(4)*250, :);
% trainTargets (si(3)*250 + 1 : si(3)*250 + si(4)*250, :) = [];

% % Run 5
% trainData = [trainData; testData];
% testData = trainData(si(3)*250 + si(4)*250 + 1 : end, :);
% trainData(si(3)*250 + si(4)*250 + 1 : end, :) = [];
% trainTargets = [trainTargets; testTargets];
% testTargets = trainTargets(si(3)*250 + si(4)*250 + 1 : end, :);
% trainTargets (si(3)*250 + si(4)*250 + 1 : end, :) = [];
