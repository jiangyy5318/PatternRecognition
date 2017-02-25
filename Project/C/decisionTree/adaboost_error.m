function e = adaboost_error(X, y, tree, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

%%% Your Code Here %%%

%%% Your Code Here %%%
%Xalpha = zeros(size(X,1),size(p,2));
N = length(y);
C = length(unique(y));
vote = zeros(C,N);
l = 1;
while(l<length(alpha)&&alpha(l)~=0)
    L_p = round(predict(tree{l},X))';
    L_p = L_p + 1;
    for i = 1:N
        vote(L_p(i),i) = vote(L_p(i),i) + alpha(l);
    end
    l = l + 1;
end
[~,pre_label] = max(vote,[],1);
pre_label = pre_label -1;
e = sum(double(pre_label~=y))/size(y,2);

end