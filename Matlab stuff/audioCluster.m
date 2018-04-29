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
Tw = 30; %frame duration in ms (from KINNUNEN clustering paper)
Ts = 10; %time shift
alpha = 0.97;
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));

R = [ 300 3700];    %range to consider
M = 20; % filterbank channels
C = 13; %cepstral coefficients
L = 22; % cepstral sine lifter parameter????


[MFCCs, FBEs, frames] = mfcc(speech, fs, Tw, Ts, alpha, hamming, R, M, C, L);

% running pca on the mfccs for plotting purposes
pcaCoeffs = pca(MFCCs');
pcaTest = MFCCs'*pcaCoeffs;
% plotting the pca
plot3(pcaTest(:,1), pcaTest(:, 2), pcaTest(:, 3), 'o', 'color', 'y');
hold on;


