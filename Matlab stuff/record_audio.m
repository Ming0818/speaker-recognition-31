% recording audio data and generating a file, given a specific path
function record_audio(f_path, secs)

recObj = audiorecorder(44100, 16, 1); % default is 9000 sample rate
% if we want to set functions to tell us when we're starting and stopping
% the recording
recObj.StartFcn = 'disp(''Start speaking.'')';
recObj.StopFcn = 'disp(''End of recording.'')';

record(recObj, secs);
pause(secs);
speech = getaudiodata(recObj);

audiowrite(f_path,speech,44100);

end
