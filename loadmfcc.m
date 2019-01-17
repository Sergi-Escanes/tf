clearvars
close all
clc

%% This script loads audio files from the computer and does pre/processing on them to output the mel frequency cepstral coefficients and the filterbank energies.

% Note: a modified version of mfcc.m called mfccs.m is used due to
% the introduction of mfcc.m by MathWorks in recent patches. The reason
% mfcc.m cannot be used is because it only allows 40 MFCCs to be extracted.
% That is insufficient for deep learning tasks, needing a total amount of
% 128 or more, commonly.

tic

N = 246;                            % number of signals per class
frameTime = 10;                     % duration of each frame (s)
inputcell = cell(N,1);
outputcell = cell(N,1);             % Initialize input and output
fs = 44100;                         % Sampling frequency

% Select MFCCs properties

    Tw = 88;                % analysis frame duration (ms)
    Ts = 44;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 227;                % number of filterbank channels 
    C = 52;                 % number of cepstral coefficients
    L = 14;                 % cepstral sine lifter parameter
    LF = 20;                % lower frequency limit (Hz)
    HF = 20000;             % upper frequency limit (Hz)

% Initialize matrix containing signal frames as columns
frameMatrix = nan(fs*frameTime,1);    
    
% Initialize routine for reading audio files from directory
for i = 1:N
   
    % Read files
    [inputcell{i}, ~] = audioread(['D:\Ev-Set(wavs)\Truck\Truck (' num2str(i) ').wav']);
    
    speech = inputcell{i};
    
    % cut signal to get desired duration for all of them
                                               
    cut = frameTime*fs - length(speech);
    
    if cut <= 0
        speech = speech(1:length(speech) + cut);
    else
        speech = padarray(speech,cut,'post');
    end
                                               
                                               
    speechn = speech / max(speech);            % normalise signal
    
    ind = 1;
    nframes = length(speechn) / (fs*frameTime);        % 1 frame
    lengthframe = frameTime*fs;                        % samples per frame

    % Initialize computation of MFCCs per frame 
    for k = 1:nframes
        
        % Frame the signal and store it in frameMatrix
        speechnframe = speechn(ind:ind+lengthframe-1);
        ind = ind + lengthframe;
        frameMatrix(:,k) = speechnframe;
    
        % Compute MFCCs and FBEs using the funtion mfcc.m
        
    [MFCCs,FBEs,frames] = ...
    mfccs(frameMatrix(:,k),fs,Tw,Ts,alpha,@hamming,[LF HF],M,C,L);
    
    [Nw,NF] = size(frames);                     % frame length and number of frames
    time_frames = (0:NF-1)*Ts*0.001+0.5*Nw/fs;  % time vector (s) for frames 
    time = (0:length(speechn)-1)/fs;            % time vector (s) for signal samples 

    meanv = mean(FBEs);
    devv = std(FBEs);
    FBEsmean = FBEs - meanv;
    FBEsdef = FBEsmean./ devv;            % standarisation
    
    % Store desired values (in this case FBEs, but same for MFCCs) in
    % ouptut cell
    
    outputcell{k} = FBEsdef;
    
    % Write numeric values as an image

    filename = ['FBE Truck (' num2str(i) ').png'];

    imwrite(outputcell{k},filename,'WriteMode','overwrite');
    
    % Step necessary for getting RGB channels, often demanded.
    
    % Read image 
    I = imread(filename);
    
    % Set color map
    map = gray(255);
    
    % Convert to RGB (get 3 channels instead of 1)
    RGB = ind2rgb(I,map);

    % Write image again
    imwrite(RGB,filename,'WriteMode','overwrite');


    end
    
 end
 
toc
 

    

    



  

