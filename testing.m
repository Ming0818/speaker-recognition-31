close all;

[audioIn, fs] = audioread('lindsay.wav');
twoStart = 110e3;
twoStop = 135e3;
audioIn = audioIn(twoStart:twoStop);
timeVector = linspace((twoStart/fs),(twoStop/fs),numel(audioIn));
figure();
plot(timeVector,audioIn);
axis([(twoStart/fs) (twoStop/fs) -1 1]);
ylabel('Amplitude');
xlabel('Time (s)');
title('Utterance - Two')
sound(audioIn,fs);



%% Pitch Identification

pD = audiopluginexample.SpeechPitchDetector;
[~,pitch] = process(pD,audioIn);

figure;
subplot(2,1,1);
plot(timeVector,audioIn);
axis([(110e3/fs) (135e3/fs) -1 1])
ylabel('Amplitude')
xlabel('Time (s)')
title('Utterance - Two')

subplot(2,1,2)
plot(timeVector,pitch,'*')
axis([(110e3/fs) (135e3/fs) 80 140])
ylabel('Pitch (Hz)')
xlabel('Time (s)');
title('Pitch Contour');

