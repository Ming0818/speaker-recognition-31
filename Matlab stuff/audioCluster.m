% will return the mfccs for 1 second of recorded voice data, and plot the
% first three principal components from pca 
function MFCCs = audioCluster(recObj)

% record your voice for 1 second 
record(recObj, 1);

pause(1);

% uncomment if you want to hear yourself (lol)
% play(recObj); 
% pause(1); 

% Store data in double-precision array.
speech = getaudiodata(recObj);
fs = 8000; % default from recorder object is 8000

% uncomment to plot the waveform
% plot(speech);

speech = speech(:,1);

[pitch, MFCC] = computePitchMFCC(speech,fs);
% Tw = 30; %frame duration in ms (from KINNUNEN clustering paper)
% Ts = 10; %time shift
% alpha = 0.97;
% hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));
% 
% R = [ 300 3700];    %range to consider
% M = 20; % filterbank channels
% C = 13; %cepstral coefficients
% L = 22; % cepstral sine lifter parameter????
% 
% 
% [MFCCs, FBEs, frames] = mfcc(speech, fs, Tw, Ts, alpha, hamming, R, M, C, L);
s = size(MFCC);
s = s(2);

MFCCs =vertcat(pitch(1:s)', MFCC);

MFCCs = (MFCCs-mean(MFCCs))./std(MFCCs);
MFCCs = MFCCs';

% running pca on the mfccs for plotting purposes
pcaCoeffs = pca(MFCCs);
pcaTest = MFCCs*pcaCoeffs;
% disp(pcaTest);
% plotting the pca
s2 = size(pcaCoeffs);
s2 = s2(2);
if s2 >= 3
    plot3(pcaTest(:,1), pcaTest(:, 2), pcaTest(:, 3), 'o', 'color', 'r');
    
end
hold on;
disp(size(MFCCs));
end
function [pitch1, mfcc1] = computePitchMFCC(x,fs)


pwrThreshold = -50; % Frames with power below this threshold (in dB) are likely to be silence
freqThreshold = 1000; % Frames with zero crossing rate above this threshold (in Hz) are likely to be silence or unvoiced speech

% Audio data will be divided into frames of 30 ms with 75% overlap
frameTime = 30e-3;
samplesPerFrame = floor(frameTime*fs);
startIdx = 1;
stopIdx = samplesPerFrame;
increment = floor(0.25*samplesPerFrame);
overlapLength = samplesPerFrame - increment;

[pitch1,~] = pitch(x,fs, ...
    'WindowLength',samplesPerFrame, ...
    'OverlapLength',overlapLength);

Tw = 30; %frame duration in ms (from KINNUNEN clustering paper)
Ts = 10; %time shift
alpha = 0.97;
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));

R = [ 300 3700];    %range to consider
M = 20; % filterbank channels
C = 13; %cepstral coefficients
L = 22; % cepstral sine lifter parameter????


[mfcc1, FBEs, frames] = mfcc(x, fs, Tw, Ts, alpha, hamming, R, M, C, L);

% mfcc1 = mfcc(x,fs,'WindowLength',samplesPerFrame, ...
%     'OverlapLength',overlapLength, 'LogEnergy', 'Replace');
% numFrames = length(pitch1);
% voicing = zeros(numFrames,1);
% 
%     for i = 1: numFrames
%         
%         xFrame = x(startIdx:stopIdx,1); % 30ms frame
% 
%         if audiopluginexample.SpeechPitchDetector.isVoicedSpeech(xFrame,fs,... % Determining if the frame is voiced speech
%                 pwrThreshold,freqThreshold)
%             voicing(i) = 1;
%         end
%         startIdx = startIdx + increment;
%         stopIdx = stopIdx + increment;
% 
%     
%     end
%     
% pitch1(voicing == 0) = nan;
% mfcc1(voicing == 0,:) = nan;


end


