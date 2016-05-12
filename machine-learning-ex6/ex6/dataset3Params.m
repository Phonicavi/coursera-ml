function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
opt_C = 1;
opt_sigma = 0.3;
max_score = 1;

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

possi = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]';

for i=1:8
    for j=1:8
        sigma = possi(i);
        C = possi(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        this_score = mean(double(predictions ~= yval));
%        display(sigma);
%        display(C);
%        display(this_score);
        if this_score < max_score
            opt_C = C;
            opt_sigma = sigma;
            max_score = this_score;
        end;
    end;
end

C = opt_C;
sigma = opt_sigma;

% =========================================================================

end