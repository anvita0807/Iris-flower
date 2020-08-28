function [p] = predictOneVsAll (X,all_theta)
  m = size(X, 1);
  p = zeros(m,1);
  h = sigmoid(X*all_theta');
  [max_val,max_ind] = max(h,[],2);
  p = max_ind;
end