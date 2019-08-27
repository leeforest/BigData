function e=error_calculate(Y,L)
count=0;
for i=1:1000
    if Y(i)==L(i), count=count+1; end
end
e=100-(count/1000*100);

