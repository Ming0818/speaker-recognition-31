% will return the mfccs for 1 second of recorded voice data, and plot the
% first three principal components from pca 
% argument plot to say whether or not it should be plotted
function MFCCs = audioCluster(recObj, to_plot)

% record your voice for 1 second 
% record(recObj, 1);
% 
% pause(1);

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


[MFCCs, FBEs, frames] = mfcc2(speech, fs, Tw, Ts, alpha, hamming, R, M, C, L);

MFCCs = (MFCCs-mean(MFCCs))./std(MFCCs);


if to_plot == true
    % running pca on the mfccs for plotting purposes
    pcaCoeffs = pca(MFCCs');
    
    % taking the mean
    mfcc_mean = nanmean(MFCCs,2);

    % running pca on the mfccs for plotting purposes
    new_mean = mfcc_mean'*pcaCoeffs;
    plot3(new_mean(:,1), new_mean(:, 2), new_mean(:, 3), 'o', 'color', 'k');
end

disp(size(MFCCs));
end