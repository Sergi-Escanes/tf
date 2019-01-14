% Project Results

% 10-fold CV setup
% ci = 1 std dev.
% time in seconds
% mean accuracy compared to classifying with ImageNet (55% acc)
% bayesian optimization on initial learning rate and L2
% decision trees used for classifying in feature learning
% they were tuned with grid search for 30s.
% decision trees were used for giving best results in less time

% Tansfer Learning

models = {'Scratch','TL04','TL19','TL34','TL48','TL63'};

time_test_tl = [6080,3190,2242,2246,2209,2394]';
timer_test_tl = time_test_tl/min(time_test_tl);
mean_test_tl = [34.16,39.90,39.31,38.52,36.93,34.22]';
std_dev_test_tl = [1.99,3.17,2.52,2.16,2.25,1.72]';

tl_test = [timer_test_tl, mean_test_tl, std_dev_test_tl];

close all
fig = figure;
for i = 1:length(mean_test_tl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Transfer Learning Test'})
    scatter(tl_test(i,1),tl_test(i,2),200,'filled')
    hold on
end
legend(models)
saveas(fig,'TL_test')

% c = linspace(1,100,length(mean_test));
% k = 1;
% ci = k*std_dev_test;
% sz = pi*ci.^2;

time_ev_tl = [5205,2626,2250,1882,1903,1525]';
timer_ev_tl = time_ev_tl/min(time_ev_tl);
mean_ev_tl = [30.16,35.10,34.06,34.27,33.12,29.95]';
std_dev_ev_tl = [0.72,0.74,2.22,0.95,1.10,0.87]';

tl_ev = [timer_ev_tl, mean_ev_tl, std_dev_ev_tl];


close all
fig = figure;
for i = 1:length(mean_ev_tl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Transfer Learning Evaluation'})
    scatter(tl_ev(i,1),tl_ev(i,2),200,'filled')
    hold on
end
legend(models)
saveas(fig,'TL_ev')

% Feature Learning

% We use the same models as before, but extract features at 
% the last pooling layer

time_test_fl = [6160,3079,2446,2173,2150,2415]';
timer_test_fl = time_test_fl/min(time_test_fl);
mean_test_fl = [32.97,39.31,37.81,36.54,36.28,32.24]';
std_dev_test_fl = [2.48,1.19,1.88,2.12,1.76,1.89]';

fl_test = [timer_test_fl, mean_test_fl, std_dev_test_fl];

close all
fig = figure;
for i = 1:length(mean_test_fl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Feature Learning Test'})
    scatter(fl_test(i,1),fl_test(i,2),200,'filled')
    hold on
end
legend(models)
saveas(fig,'FL_test')

time_ev_fl = [5601,3304,2585,2401,2355,2002]';
timer_ev_fl = time_ev_fl/min(time_ev_fl);
mean_ev_fl = [28.59,34.53,34.32,33.41,32.47,28.68]';
std_dev_ev_fl = [1.15,1.09,1.04,0.65,0.66,0.99]';

fl_ev = [timer_ev_fl, mean_ev_fl, std_dev_ev_fl];

close all
fig = figure;
for i = 1:length(mean_ev_fl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Feature Learning Evaluation'})
    scatter(fl_ev(i,1),fl_ev(i,2),200,'filled')
    hold on
end
legend(models)
saveas(fig,'FL_ev')

close all

% Plot TL all

tl_all = [tl_test tl_ev];

close all
fig = figure;
for i = 1:length(mean_ev_fl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Transfer Learning Test vs Evaluation'});
    scatter(tl_all(i,1 & 4)/55,tl_all(i,2 & 5)/55,200,'filled')
    hold on
end
legend(models)
% legend(models)
hold on
for i = 1:length(mean_ev_tl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Transfer Learning Evaluation'})
    scatter(tl_ev(i,1),tl_ev(i,2)/55,200,'filled')
    hold on
end
legend(models)

saveas(fig,'TL_all')

% Plot FL all

close all
fig = figure;
for i = 1:length(mean_ev_fl)
    xlabel({'Relative Prediction Time on GPU'});
    ylabel({'Accuracy (%)'});
    title({'Feature Learning Test vs Evaluation'})
    scatter(fl_test(i,1),fl_test(i,2)/55,200,'filled')
    hold on
    scatter(fl_ev(i,1),fl_ev(i,2)/55,200,'filled')
end
legend(models)
legend(models)
saveas(fig,'FL_all')


%% t-test

N = 10;
t = nan(length(mean_test_tl),length(mean_test_tl));
v = nan(length(mean_test_tl),length(mean_test_tl));
x = nan(length(mean_test_tl),length(mean_test_tl)); % 6x6 matrices

crit = 1.833; % at 0.95 confidence level and 9 degrees of freedom
% from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm
    
% degrees of freedom
% v = ((std_dev_test_tl(1)^2 + std_dev_test_tl(2)^2)/N^2)/(((std_dev_test_tl(1)/N)^2)/(N-1) + ((std_dev_test_tl(2)/N)^2)/(N-1));

% v = 9;

for j = 1:length(mean_test_tl)
    for i = 1:length(mean_test_tl)
        t(j,i) = abs((mean_ev_fl(j) - mean_ev_fl(i))/sqrt((std_dev_ev_fl(j)^2 + std_dev_ev_fl(i)^2)/10));
%       v(j,i) = ((std_dev_ev_tl(j)^2 + std_dev_ev_tl(i)^2)/N^2)/(((std_dev_ev_tl(j)/N)^2)/(N-1) + ((std_dev_ev_tl(i)/N)^2)/(N-1));
        if t(j,i) > crit
            x(j,i) = 1;
        else
            x(j,i) = 0;
        end
    end
end

x

% x_test_tl =
% 
%      0     1     1     1     1     0
%      1     0     0     0     1     1
%      1     0     0     0     1     1
%      1     0     0     0     0     1
%      1     1     1     0     0     1
%      0     1     1     1     1     0

% x_ev_tl =
% 
%      0     1     1     1     1     0
%      1     0     0     1     1     1
%      1     0     0     0     0     1
%      1     1     0     0     1     1
%      1     1     0     1     0     1
%      0     1     1     1     1     0

% x_test_fl =
% 
%      0     1     1     1     1     0
%      1     0     1     1     1     1
%      1     1     0     0     1     1
%      1     1     0     0     0     1
%      1     1     1     0     0     1
%      0     1     1     1     1     0

% x_ev_fl =
% 
%      0     1     1     1     1     0
%      1     0     0     1     1     1
%      1     0     0     1     1     1
%      1     1     1     0     1     1
%      1     1     1     1     0     1
%      0     1     1     1     1     0


close all
