function [e_train, e_test] = adaboost(X, y, X_test, y_test, maxIter)
% adaboost: carry on adaboost on the data for maxIter loops
%
% Input 
%     X       : n * p matirx, training data
%     y       : n * 1 vector, training label
%     X_test  : m * p matrix, testing data
%     y_test  : m * 1 vector, testing label
%     maxIter : number of loops
%
% Output
%     e_train : maxIter * 1 vector, errors on training data
%     e_test  : maxIter * 1 vector, errors on testing data


w = (1 / (length(y))) * ones(length(y),1); % initialize
% k = zeros(maxIter, 1);
% a = zeros(maxIter, 1);
% d = zeros(maxIter, 1);
tree = cell(maxIter,1);
alpha = zeros(maxIter, 1);

e_train = zeros(maxIter, 1);
e_test = zeros(maxIter, 1);
for i = 1:maxIter
    %Tree(i) = decision_stump(X, y, w);
    %fprintf( 'new decision stump k:%d a:%d, d:%d\n', k(i), a(i), d(i));
    tree{i} = fitctree(X, y,'weight',w);
    
    L_p = round(predict(tree{i},X))';
    e = sum(L_p~=y)/length(L_p);
    alpha(i) = log((1 - e) / e);
    w = w.*exp(alpha(i).*(L_p~= y))';
    w = w./sum(w); 
    
    e_train(i) = adaboost_error(X, y, tree, alpha);
    e_test(i) = adaboost_error(X_test, y_test, tree, alpha);
    fprintf( 'weak learner error rate: %f\nadaboost error rate: %f\ntest error rate: %f\n\n', e, e_train(i), e_test(i));
    
%     if i > 10
%         if sum(e_test(i-5:i)) > sum(e_test(i-10:i-5))
%             lastIter = i;
%             break;
%         end
%     end
end

end