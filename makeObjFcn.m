function ObjFcn = makeObjFcn(imdsTrain,imdsTest,net)
ObjFcn = @ValError;
    function [ValError,cons,bayesnet] = ValError(optVars)
        
%% This function generates an objective function that is called on cnn.m that is the one to be optimised using the routine bayesopt.

% This function takes as input training and test databases as well as an
% untrained NN and outputs and outputs the optimal variables for BO for
% both hyperparameter optimization and FE.

% Get test labels
YTest = imdsTest.Labels;
  


% Transfer Learning 

% Set training options

options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',99, ...
    'ValidationPatience',10,...
    'Verbose',false, ...
    'MiniBatchSize',16,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'InitialLearnRate',optVars.InitialLearnRate,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.1,...
    'L2Regularization',optVars.L2Regularization,...
    'Shuffle','every-epoch',...
    'GradientThreshold',Inf,...
    'GradientThresholdMethod','l2norm',...
    'Plots','none',...
    'ExecutionEnvironment', 'gpu');

CNN = trainNetwork(imdsTrain,net,options);



% Feature Extraction (FE)

% Set #layer where features are extracted

% layer = 'prob';             %layer #67
% 
    % Define features to be extracted
% featuresTrain = activations(CNN,imdsTrain,layer,'OutputAs','rows');
% featuresTest = activations(CNN,imdsTest,layer,'OutputAs','rows');

% Classify on test set using a trained CNN
YPred = classify(CNN,imdsTest); % Comment if using FE


% Feed features to the classifier and optimize its parameters
% classifier = fitcecoc(featuresTrain,YTrain,'Coding','onevsall','Learners','tree',...
%     'OptimizeHyperparameters',...
%     'all','HyperparameterOptimizationOptions',...
%     struct('Optimizer','bayesopt','MaxTime',30)); % Uncomment if using FE
% 
% Predict labels
% YPred = predict(classifier,featuresTest); % Uncomment if using FE


% Define validation error
ValError = 1 - sum(YPred == YTest)/numel(YTest);


% Store and save results
bayesnet = num2str(ValError) + ".mat";
        save(bayesnet,'CNN','ValError','options')
        cons = [];
        
    end
end