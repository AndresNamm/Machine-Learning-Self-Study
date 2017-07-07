y_vec=zeros(m,num_labels);

for i = 1:num_labels
    n = (y==i);
    y_vec(n,i)=1;
end



triangle1 = zeros(size(Theta1));
triangle2 = zeros(size(Theta2));
for i = 1:m
    % FORWARD PROPAGATION 
    a1=[1 ,X(i,:)]'; % Get 1 observation from the dataset
    z2=Theta1*a1; % Produces (#theta_rows X 1) vector
    a2=[1;sigmoid(z2)];
    z3=Theta2*a2;
    a3=sigmoid(z3);
    % dE/dz L..2
    delta3=a3-y(i,:)';%partial derivative z3 
    delta2=Theta2'*delta3.*a2.*(1-a2); %partial derivative z2
    delta2=delta2(2:end);


    % triangle = triangle + dE/dTheta   
    triangle1 = triangle1 + delta2*a1';
    triangle2 = triangle2 + delta3*a2';  
end 



