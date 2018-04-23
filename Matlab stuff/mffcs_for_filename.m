%filename = 'jake.wav'

%% Actual Script
name = char(filename);

[ speech, fs ] = audioread( name );
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



figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto','color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
imagesc( [1:size(MFCCs,2)], [0:C-1], MFCCs ); 
axis( 'xy' );
xlabel( 'Frame index' ); 
ylabel( 'Cepstrum index' );
title( 'Mel frequency cepstrum' );