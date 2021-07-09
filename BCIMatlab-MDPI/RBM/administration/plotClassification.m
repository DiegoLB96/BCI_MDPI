function [varargout] = plotClassification(rootpath, subjectExt, subjectNum, type, numFig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   created 01.12.2010 - last modified 22.02.2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot the results of the classification error or accuracy
% plot the reconstruction of the different layers
% 
if nargin < 3, error('Needs the extension the subject Extension and subject number'); end
if nargin < 4, type = 'Error'; end
if nargin < 5 || numFig < 0, numFig = 0; end

%%%%%%%%%%%%%%%%%%%%%%%%%
Data = load([rootpath 'Data' num2str(subjectNum)]);
stats = Data.dataStats;
numClasses = size(Data.trainTargets,2);
RBMData = load([rootpath subjectExt num2str(subjectNum)]);
BPtestErr = RBMData.ErrBP.testErr;
BPtrainErr = RBMData.ErrBP.trainErr;
RBMtestErr = RBMData.ErrRBM.testErr;
RBMtrainErr = RBMData.ErrRBM.trainErr;
BPerrType = RBMData.ErrBP.type;
RBMerrType = RBMData.ErrRBM.type;
RBMtestNumClasses = RBMData.ErrRBM.testNumClass;
RBMtrainNumClasses = RBMData.ErrRBM.trainNumClass;
BPtestNumClasses = RBMData.ErrBP.testNumClass;
BPtrainNumClasses = RBMData.ErrBP.trainNumClass;

if exist([rootpath 'RBMdataRec' num2str(subjectNum) '.mat'],'file')
    RBMDataRec = load([rootpath 'RBMdataRec' num2str(subjectNum) '.mat']);
    reconst = RBMDataRec.reconst;
    averageSignal = RBMDataRec.averageSignal;
    if isfield(stats,'Fv'),
        fv = stats.Fv; % not in all cases there is a frequency vector 
    else
        fv = 1:size(reconst,2);
    end
    clear RBMDataRec
end

clear RBMData Data

%%%% ADMINISTRATION %%%%
colors = [1 0 0;0 1 0;0 0 1;0 0 0; 0 1 1];
set(0,'DefaultLineLinewidth',3)
if numClasses < size(colors,1)
set(0,'DefaultAxesFontSize',18,'DefaultAxesColorOrder',colors(1:numClasses,:),...
      'DefaultAxesLineStyleOrder','-|--|:')
else
    set(0,'DefaultAxesFontSize',18,'DefaultAxesColorOrder',colors,...
      'DefaultAxesLineStyleOrder','-|--|:')
end
titulo = ['Subject ' subjectExt num2str(subjectNum) ', RBM ' type];
txt = [stats.Type 'T:' num2str(stats.T) ', fs:' num2str(stats.fs)];
txt = {txt, sprintf(stats.Coments)};
%%% RBM error plot %%%
epochs = 1:size(RBMtrainErr,1);
numFig = numFig + 1;
h = plotting(epochs, numFig, RBMtestErr, RBMtrainErr, RBMerrType, titulo, txt, type, RBMtrainNumClasses, RBMtestNumClasses);
saveas2(h, [rootpath subjectExt num2str(subjectNum) 'RBM' '.pdf'], 300, 'pdf')

%%% Back propagation error plot %%%
epochs = 1:size(BPtrainErr,1);
numFig = numFig + 1;
h = plotting(epochs, numFig, BPtestErr, BPtrainErr, BPerrType, titulo, txt, type, BPtrainNumClasses, BPtestNumClasses);
saveas2(h, [rootpath subjectExt num2str(subjectNum) 'BackPropagation' '.pdf'], 300, 'pdf')

%%% plot Reconstruction %%%
if exist([rootpath 'RBMdataRec' num2str(subjectNum) '.mat'],'file')
    numFig = numFig + 1;
    h = figure(numFig);    
    set(h,'units','normalized','outerposition',[0 0 1 1]);
    for j = 1 : numClasses
        sizeChannel = size(fv);
        
        subplot(numClasses,3,1+3*(j-1))
        semilogx(fv,reconst(j,1:sizeChannel)), hold on
        semilogx(fv,averageSignal(j,1:sizeChannel),'-.b')
        xlim([8 30])
        ylim([-1 1 ])
        if j == numClasses
            set(gca,'XTick',[9 10 20 30])
            set(gca,'XTickLabel',{'', '10', '20', '30'})
%             xlabel('Frequency [Hz]')
        else 
            set(gca,'xticklabel',[])
        end        
        if j == 2
            title('Cz')
        end
        grid on
        
        subplot(numClasses,3,2+3*(j-1))
        semilogx(fv,reconst(j,sizeChannel+1:2*sizeChannel)), hold on
        semilogx(fv,averageSignal(j,sizeChannel+1:2*sizeChannel),'-.b')
        xlim([8 30])
        ylim([-1 1 ])
        if j == numClasses
            set(gca,'XTick',[9 10 20 30])
            set(gca,'XTickLabel',{'', '10', '20', '30'})
            xlabel('Frequency [Hz]')
        else 
            set(gca,'xticklabel',[])
        end  
        set(gca,'yticklabel',[])
        if j == 2
            title('C3')
        end
        grid on
        
        subplot(numClasses,3,3+3*(j-1))
        semilogx(fv,reconst(j,2*sizeChannel+1:3*sizeChannel)), hold on
        semilogx(fv,averageSignal(j,2*sizeChannel+1:3*sizeChannel),'-.b')
        xlim([8 30])
        ylim([-1 1 ])     
        if j == numClasses
            set(gca,'XTick',[9 10 20 30])
            set(gca,'XTickLabel',{'', '10', '20', '30'})
%             xlabel('Frequency [Hz]')
        else 
            set(gca,'xticklabel',[])
        end  
        if j == 2
            title('C4')
        end
        grid on
    end
    saveas2(h, [rootpath subjectExt num2str(subjectNum) 'Recontruction' '.pdf'], 300, 'pdf')
end
varargout = {numFig};
varargout =  varargout(1:nargout);

end

function h = plotting(epochs, numFig, testErr, trainErr, errType, titulo, txt, type, RBMtrainNumClasses, RBMtestNumClasses)
 
if ~strcmp(type, 'Error')
    testErr = 1 - testErr;
    trainErr = 1 - trainErr;
end
h = figure(numFig);
    set(h,'units','normalized','outerposition',[0 0 1 1]);
    if nargin > 8
        subplot(3,1,1)
        plot(epochs,RBMtrainNumClasses,epochs,RBMtestNumClasses)
        leyenda = strcat(repmat('Class: ',size(RBMtrainNumClasses,2),1), num2str([1:size(RBMtrainNumClasses,2)]'));
        legend( leyenda )
        title('Percentage of classes')
        ylabel('[%] per Class')
		ylim([0 1])
        set(gca,'xtick',[])
        subplot(3,1,[2 3])
    end
    plot(epochs,trainErr,epochs,testErr)
    title(titulo)
    ylabel([type '[%]'])
    ylim([0.5 1])
    xlabel(['# Epochs,' errType])
    legend('Training','Testing')
    a = axis;
    wdth = a(2)-a(1);
    ht = a(4)-a(3);
    pos = [a(1)+0.1*wdth a(4)-0.1*ht];
    text(pos(1),pos(2), txt);
%     extraPlot = axes('Position',[.7 .5 .15 .15],'Visible','off');

    
drawnow
end
