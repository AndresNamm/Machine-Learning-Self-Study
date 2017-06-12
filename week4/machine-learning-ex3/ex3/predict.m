function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
% Theta1 has size 25 x 401 
% Theta2 has size 10 x 26
% Useful values

m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 

X = [ ones(m,1) , X ];
p = zeros(size(X, 1), 1);
hidden = sigmoid(Theta1*X'); % hidden 25 x 5000
hidden = [ones(1,m); hidden]; % hidden 26 x 5000
hx = sigmoid(Theta2*hidden); % hx 10 x 5000
hx = hx'; % 5000 x 10 
[var pos] = max(hx, [], 2);
p = pos; 






% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
