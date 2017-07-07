y_vec=zeros(m,num_labels);

for i = 1:num_labels
    n = (y==i);
    y_vec(n,i)=1;
end
