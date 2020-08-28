function [all_theta] = oneVsAll (X, y, num_of_classes, lambda)
  m = size(X, 1);
  n = size(X, 2);
  all_theta = zeros(num_of_classes, n);
  
  % for this example we have 3 classes
  
  for i = 1: num_of_classes
    % Set the parameters for optimization
    % ==================================
      initial_theta = zeros(n, 1);    
      options = optimset('GradObj','on','MaxIter',50);
    % ==================================
    costfunc = @(t)(lrCostFunction(t,X,y == i,lambda));
    [theta] = fmincg(costfunc,initial_theta,options);
    % (y == i)divides the training set into 2  classes
    
    
    all_theta(i,:) = theta; % 3x5 matrix
    
    end
end