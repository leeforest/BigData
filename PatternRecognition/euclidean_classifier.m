function Y=euclidean_classifier(X,m)

Y=[];
for i=1:1000
    for j=1:3
        y(j)=sqrt((X(:,i)-m(:,j))'*(X(:,i)-m(:,j)));
    end
 [n, Y(i)]=min(y);
end
end

