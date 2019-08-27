function Y=bayes_classifier(X,m,S,P)

Y=[];
y=[];
for i=1:1000
    for j=1:3
        y(j)=P(j)*(1/((2*pi)^(1/2)*det(S)^0.5))*exp(-0.5*(X(:,i)-m(:,j))'*inv(S)*(X(:,i)-m(:,j))); 
    end
 [n, Y(i)]=max(y);
end
end

