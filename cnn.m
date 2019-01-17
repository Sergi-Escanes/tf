clearvars
close all
% clc

%% This script trains several deep learning models using Transfer Learning (TL) and Feature Learning (FL) on an image database.

% The model used is SqueezeNet, a model that achieves same accuracy (0.55)
% as AlexNet on ImageNet but uses 60 times less parameters.

% The database is a collection of images that represent spectrograms from a
% subset of AudioSet related to smart cars.

% The script includes the functions makeObjFcn.m and loadmffc.m. The first
% contains what is necessary for performing Feature Extraction (FE) for FL
% as well as Bayesian Optimization, used for hyperparameter optimization
% and FL. The second contains the process of converting sound samples to
% spectrograms.

%% CNN

DatasetPathDev = fullfile('D:\','Dev-Set-Rus(fbes)'); 

DatasetPathEv = fullfile('D:\','Ev-Set(fbes)'); % File directories

% Create Image Databases

% Development Database

imdsDev = imageDatastore(DatasetPathDev, ...
    'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

% Evaluation Database

imdsEv = imageDatastore(DatasetPathEv, ...
    'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

% Plot database examples

% figure;
% perm = randperm(208,4);
% for i = 1:4
%     subplot(2,2,i);
%     imshow(imdsDev.Files{perm(i)});
% end

% labelCount = countEachLabel(imdsDev);

nfolds = 10; % # CV folds
cv = cvpartition(imdsDev.Labels,'KFold',nfolds,'Stratify',false); % generate cv folds
accuracy = nan(nfolds,1); % initialize accuracy vector

% Load TL models to workspace

load squeezenet_scratch
load squeezenet_TL04
load squeezenet_TL19
load squeezenet_TL34
load squeezenet_TL48
load squeezenet_TL63

nets = [squeezenet_scratch,squeezenet_TL04,squeezenet_TL19,squeezenet_TL34,squeezenet_TL48,squeezenet_TL63];

% Define L2 and Initial Learning Rate hyperparameters

L2optmean = 0.0325;
ILRoptmean = 4.0818e-5;

% layer = 'prob';     % FE layer #67


% Initialize training routine for all models

for i = 1:length(nets)
tic
net = nets(i);

% Hyperparameters to be optimized

% optimVars = [
%     optimizableVariable('L2Regularization',[1e-3 1e-1],'Transform','log')
%     optimizableVariable('InitialLearnRate',[1e-5 1e-3],'Transform','log')
%     ];
    
    % Initialize training routine per fold
   
    for k = 1:nfolds
    
    % Reset GPU values (not necessary, but maybe helpful)
    g = gpuDevice(1);
    reset(g);
    
    % Get training and test files for cv partition
    Train = imdsDev.Files(cv.training(k));
%     Test = imdsDev.Files(cv.test(k));
    
    % Re-generate image databases for cv partition (training)
    imdsTrain = imageDatastore(Train,'LabelSource','foldernames');
    
    % Resize images for training set
    % augimdsTrain = augmentedImageDatastore([224 224],imdsTrain);

    % Set training labels
    YTrain = imdsTrain.Labels;

    % Re-generate image databases for cv partition (test)
%     imdsTest = imageDatastore(Test,'LabelSource','foldernames');

    % Resize images for training set
    % augimdsTest = augmentedImageDatastore([224 224],imdsTest);

    % Set test labels
%     YTest = imdsTest.Labels;  % Uncomment if Test
    
    % Set evaluation labels
    YEv = imdsEv.Labels;    % Uncomment if Ev

    % Call ObjFcn.m for Bayesian Optimization (BO) on L2 and IRL parameters
    
%     ObjFcn = makeObjFcn(imdsTrain,imdsTest,net);
%      
%     BayesObject = bayesopt(ObjFcn,optimVars,...
%         'MaxObj',30,...
%         'MaxTime',8*60*60,...
%         'IsObjectiveDeterministic',false,...
%         'UseParallel',false);
%     
%     close all

    % Set training options
    
    options = trainingOptions('adam', ... % Comment if using BO
        'MaxEpochs',30, ...
        'ValidationData',imdsEv, ...
        'ValidationFrequency',99, ...
        'ValidationPatience',10,...
        'Verbose',false, ...
        'MiniBatchSize',16,...
        'SquaredGradientDecayFactor',0.999,...
        'Epsilon',1e-8,...
        'InitialLearnRate',ILRoptmean,... 
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',10,...
        'LearnRateDropFactor',0.1,...
        'L2Regularization',L2optmean,... 
        'Shuffle','every-epoch',...
        'GradientThreshold',Inf,...
        'GradientThresholdMethod','l2norm',...
        'Plots','none',...
        'ExecutionEnvironment', 'gpu');
    
    % Train the CNN
 
    CNN = trainNetwork(imdsTrain,net,options); % Comment if using BO
 
    % Classify using trained CNN on test or evaluation sets
    
    YPred = classify(CNN,imdsEv); % Comment if using FE or BO

    % Feature Extraction

      % Define extracted features
      
%     featuresTrain = activations(CNN,imdsTrain,layer,'OutputAs','rows');
%     featuresTest = activations(CNN,imdsEv,layer,'OutputAs','rows');
% 
     % Feed the features to the classifier (decision trees in this case)
     % and optimize its parameters
     
%     classifier = fitcecoc(featuresTrain,YTrain,'Coding','onevsall','Learners','tree',...
%         'OptimizeHyperparameters',...
%         'all','HyperparameterOptimizationOptions',...
%         struct('Optimizer','gridsearch','MaxTime',30)); % Uncomment if using FE
% 
    % Classify using FE
%     YPred = predict(classifier,featuresTest); % Uncomment if using FE

    % Call best parameters for Bayesian Optimization on the classifier

%     bestIdx = BayesObject.IndexOfMinimumTrace(end);
%     fileName = BayesObject.UserDataTrace{bestIdx};
%     savedStruct = load(fileName);
%     ValError = savedStruct.ValError;
%     
%     [YPred,probs] = classify(savedStruct.CNN,imdsTest);

    % Compute accuracy
    accuracy(k) = sum(YPred == YEv)/numel(YEv);

    
    end

    % Set confidence intervals
pd = fitdist(accuracy,'Normal');
ci = paramci(pd);

% Print mean accuracy and mean+/-std dev
Acc = [pd.mu, ci(1,1) - pd.mu, ci(2,1) - pd.mu]


% Plot confusion matrix
% plotconfusion(YTest,YPred)

toc

end

close all
