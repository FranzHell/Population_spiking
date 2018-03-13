clear
load ('data/population_data1.mat')
%% signal and noise correlations

% there is a 1 s silent period before the start of each stimulus,
% the stimulus lasts 0.5 s, trial periode is 2 s:
%
% __________-----_____
%
% data was recorded contineously, so there was no gap between single trials 

% analysis parameters - all units are in s
stimStart   = 1.015; % start of stimulus + 15 ms latency
stimEnd     = 1.515; % end of the stimulus + 15 ms latency
win         = [stimStart stimEnd]; % evaluation window for tuning, signal and noise correlations 

trials = unique(trialNo);       % get all possible unique trial numbers
trials(trials == 0) = [];       % get rid of the zeros
nTrials = length(trials);       % number of trials  
nUnits = size(spikeTimes,2);    % number of units in data
f = unique(freqList);           % frequncie
f(f==0) = [];                   % get rid of zeros 
nFreqs = length(f);             % number of presented frequencies
nReps = ceil(nTrials/nFreqs);   % number of repetions of each stimulus



%% spike counts in the stimulus window
% in order to get tuning of the cells we need to count the spikes in the
% stimulus window and order counts by stimulus frequency

spikeMask = spikeTimes > win(1) & spikeTimes <= win(2); % 1 for spikes in window, 0 for all others
spikeNumber = zeros(nFreqs,nReps,nUnits); % we'll collect spike counts in a 3D array: freqs x trials x units

for iF = 1:nFreqs % loop through stimulus frequencies
    thisTrials = trials(freqList == f(iF)); 
    for iRep = 1:length(thisTrials) % loop through repetitions
        spikeNumber(iF,iRep,:) = sum(spikeMask & trialNo == thisTrials(iRep)); % sum only spikes from current trial
    end
end

%% tuning curves - mean rates
% Plot tuning curves for all units in subplots

spikeRate = spikeNumber/diff(win);    %  divided by window duration to get Hz
tuning = squeeze(mean(spikeRate,2));  %  mean over all repetions 

tuningFig = figure;
for iU=1:nUnits
    subplot(4,5,iU)
    semilogx(1e-3*f,tuning(:,iU)) % frequency in kHz
    title(sprintf('Unit %d',iU))
    axis tight
    if iU == 18
        xlabel('frequency [kHz]')
    elseif iU == 11
        ylabel('spike rate [Hz]')
    end
    
end

%% signal correlations

% get signal correlations for all units
signalCorr = corrcoef(tuning);

% plot mean responses for pairs 9/13 and 8/12
xmpls = [1,4;8,12];
exampleFig = figure;
for iX = 1:2
    subplot(2,2,iX)
    scatter(tuning(:,xmpls(iX,1)),tuning(:,xmpls(iX,2)))
    xlabel('rate cell 1 [Hz]')
    ylabel('rate cell 2 [Hz]')
    title(sprintf('Signal Correlation, r=%0.2g',signalCorr(xmpls(iX,1),xmpls(iX,2))))
end

% get all values below diagonal
signalCorrVect = signalCorr(logical(tril(ones(nUnits),-1)));

%% noise correlations

% normalize 
normSpRate = bsxfun(@rdivide, spikeRate, reshape(max(tuning),[1,1,nUnits]));
normSpRate = spikeRate./ reshape(max(tuning),[1,1,nUnits]);
% subtract mean tuning
responseNoise = bsxfun(@minus,normSpRate,mean(normSpRate,2));

% reshape keeping repetions of all units aligned
responseNoise = reshape(responseNoise,[nFreqs*nReps,nUnits]);
% responseNoise = squeeze(mean(responseNoise,2))
% calculate noise correlations
noiseCorr = corrcoef(responseNoise);

% get all values below diagonal
noiseCorrVect = noiseCorr(logical(tril(ones(nUnits),-1)));
figure
subplot(2,1,1)
imagesc(noiseCorr)
subplot(2,1,2)
imagesc(signalCorr)
% plot noise correlations for the examples above (same figure)
figure(exampleFig);
for iX = 1:2
    subplot(2,2,iX+2)
    scatter(responseNoise(:,xmpls(iX,1)),responseNoise(:,xmpls(iX,2)))
    xlabel('reponse deviation cell 1')
    ylabel('response cell 2')
    title(sprintf('Noise Correlation, r=%0.2g',noiseCorr(xmpls(iX,1),xmpls(iX,2))))
end

% compare noise and signal correlations - is there any trend?
corrCompFig = figure;
scatter(signalCorrVect,noiseCorrVect)
xlabel('Signal correlations')
ylabel('Noise correlations')
lsline
corrcoef(signalCorrVect,noiseCorrVect)

%% classification of responses (optional)

% we will simply use a distance to the populatin vector - we will use the
% normalized rate 'normSpRate'

normTuning = squeeze(mean(normSpRate,2));

% now run through  population responses to all frequencies and simply classify as the
% nearest to the mean response for the 
edges = (0:nFreqs)+0.5; % bin edges for use with histcounts
confMtrx = zeros(nFreqs);
for iF = 1:nFreqs
    [~,idx] = min(pdist2(squeeze(normSpRate(iF,:,:)),normTuning),[],2); % minimal distance
    confMtrx(:,iF) = histcounts(idx,edges,'normalization','probability');
end

percCorr = 100*mean(confMtrx(~~eye(nFreqs))); % diagonal elements are classificaton success
confFig = figure;
imagesc(confMtrx) % plot confusion matrix
title(sprintf('classification (%0.1f %% correct)', percCorr))
xlabel('stimulus played')
ylabel('stimulus classified')
cBar = colorbar;
cBar.Label.String = 'classification probability';


%% plot responses to silence as dot plots to see coordination of activity in silence

% take only the first 100 trials (20 s of silence)
nTrials = 100; % silence is 1s, so 20 seconds of silence
binSize = 0.05; % seconds

sT = spikeTimes;
firstTrials = trials(1:nTrials);
sT(trialNo>firstTrials(end)) = 0;
edges =  binSize:binSize:stimStart*nTrials; % bins for spike counts in histcounts 
edges = edges + (firstTrials(1)-1)*stimStart; % need to correct this because trials do not need to start at the first trial
nBins = length(edges)-1; % number of bins

binnedResp = zeros(nBins,nUnits);

dotPlotFig = figure;
for iU = 1:nUnits
    thisIdcs = sT(:,iU) > 0 & sT(:,iU) < stimStart ; % take only spikes in the silent period before the stimulus
    thisTrialNo = trialNo(thisIdcs,iU); 
    thisTimes = spikeTimes(thisIdcs,iU) + stimStart*(thisTrialNo-1);
    
    % get spike counts in bins over time
    binnedResp(:,iU) = histcounts(thisTimes,edges);
    
    % plot lines (slower but much nicer than just scatter plots)
    x = thisTimes(thisTimes<(10+edges(1)))'; % do not use more than 10 seconds for plotting 
    y=iU*ones(1,length(x));
    line([x;x], [y;y+1],'color','k')
    xlabel('time [s]')
    ylabel('unit number')
end

% plot multi unit activity
mua = sum(binnedResp,2); % simply sum all spiking
hold on
t = edges(1:end-1)+binSize/2;
plot(t,nUnits*mua/max(mua),'r','linewidth',2);
xlim(edges(1)+[0 10])

%% correlation between multi-unit and single-unit activity during silence

muaSglCorr = zeros(nUnits,1);
for iU = 1:nUnits
    thisMUA = binnedResp;
    thisMUA(:,iU) = []; % take the MUA without the current unit
    temp = corrcoef(binnedResp(:,iU),mean(thisMUA,2));
    muaSglCorr(iU) = temp(1,2);
end

% plot a histogram in order to see wether the population seperates in
% chorussers and soloists
muaCorrFig = figure;
histogram(muaSglCorr,10)
xlabel('correlation with MUA')
ylabel('number of occurences')

