function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Values for C and for sigma
vals = [0.001 0.01 0.1 1 10]';
%vals = [10 0.001]';


% You need to return the following variables correctly

C = 1;
sigma =0.1;
%err=zeros(length(vals),length(vals));
%min = 1000000;
%for i = 1:length(vals)
%  for j = 1:length(vals)
%    C=vals(i);
%    sigma=vals(j);
%    model= svmTrain(X, y, vals(i), @(x1, x2) gaussianKernel(x1, x2, vals(j))); 
%    hval = svmPredict(model,Xval);
%    %err(i,j) = sum((hval-yval).**2);  - first idiocy
%    %err(i,j) = sum((yval'*(yval-hval)+(1-yval)'*(hval)).**2); 
%    err(i,j)= mean(double(hval ~= yval));
%    if err(i,j)<min
%       C=vals(i)X,;
%       sigma=vals(j);
%    end
%  end 
%end 
%err
    
    


    
    


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
