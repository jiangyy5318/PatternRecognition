load('../pool5.mat');
chooseTrain = randperm(size(train_X,1));
chooseTest = randperm(size(test_X,1));
maxIter = 30;
tic;
[e_trainpool5, e_testpool5] = adaboost(train_X(chooseTrain(1:1000),:), train_Y(1,chooseTrain(1:1000)), test_X(chooseTest(1:1000),:), test_Y(1,chooseTest(1:1000)), maxIter);
toc;

tic;
load('../fc6.mat')
maxIter = 30;
[e_trainfc6, e_testfc6] = adaboost(train_X(chooseTrain(1:1000),:), train_Y(1,chooseTrain(1:1000)), test_X(chooseTest(1:1000),:), test_Y(1,chooseTest(1:1000)), maxIter);
toc;
plot(1:maxIter,e_testpool5(1:maxIter),'b-',1:maxIter,e_testfc6(1:maxIter),'k-');
title('test error vs iter(pool5 and fc6)');
legend('pool5','fc6');
