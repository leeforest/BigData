
% ��� ����
m1=[1; 1];
m2=[4; 4];
m3=[8; 1];
m=[m1 m2 m3];

% ���л� ���
S=2*[1 0; 0 1];

% ���� Ȯ��
P=[0.33; 0.33; 0.34];
P_=[0.8; 0.1; 0.1];

% ù ��° �����ͼ�(X5)
x5_1=mvnrnd(m1,S,P(1)*1000);
x5_2=mvnrnd(m2,S,P(2)*1000);
x5_3=mvnrnd(m3,S,P(3)*1000);
X5=[x5_1; x5_2; x5_3];

% �� ��° �����ͼ�(X5_)
x5__1=mvnrnd(m1,S,P_(1)*1000);
x5__2=mvnrnd(m2,S,P_(2)*1000);
x5__3=mvnrnd(m3,S,P_(3)*1000);
X5_=[x5__1; x5__2; x5__3];

% ������ �з��� ����
Y1=bayes_classifier(X5',m,S,P);
Y2=bayes_classifier(X5_',m,S,P);

% ������ �з� ���(X5)
result1=[X5, Y1'];
figure(1);
hold on;
for i=1:1000
    t=X5(:,1);
    if result1(i,3)==1, scatter(result1(i,1),result1(i,2),'r'); end
    if result1(i,3)==2, scatter(result1(i,1),result1(i,2),'g'); end
    if result1(i,3)==3, scatter(result1(i,1),result1(i,2),'b'); end
end

% ������ �з� ���(X5_)
result2=[X5_, Y2'];
figure(2);
hold on;
for i=1:1000
    t=X5_(:,1);
    if result2(i,3)==1, scatter(result2(i,1),result2(i,2),'r'); end
    if result2(i,3)==2, scatter(result2(i,1),result2(i,2),'g'); end
    if result2(i,3)==3, scatter(result2(i,1),result2(i,2),'b'); end
end

% ��Ŭ����� �з��� ����
Y3=euclidean_classifier(X5',m);
Y4=euclidean_classifier(X5_',m);

% ��Ŭ����� �з� ���(X5)
result3=[X5, Y3'];
figure(3);
hold on;
for i=1:1000
    t=X5(:,1);
    if result3(i,3)==1, scatter(result3(i,1),result3(i,2),'r'); end
    if result3(i,3)==2, scatter(result3(i,1),result3(i,2),'g'); end
    if result3(i,3)==3, scatter(result3(i,1),result3(i,2),'b'); end
end

% ��Ŭ����� �з� ���(X5_)
result4=[X5_, Y4'];
figure(4);
hold on;
for i=1:1000
    t=X5_(:,1);
    if result4(i,3)==1, scatter(result4(i,1),result4(i,2),'r'); end
    if result4(i,3)==2, scatter(result4(i,1),result4(i,2),'g'); end
    if result4(i,3)==3, scatter(result4(i,1),result4(i,2),'b'); end
end

% ���� ����� ���� ��(L1) ����(X5)
label1_1=ones(1,330)';
label1_2=2*label1_1;
label1_3=3*label1_1;
L1=[label1_1;label1_2; label1_3];
for i=1:10
    L1(end+1)=3;
end

% ���� ����� ���� ��(L2) ����(X5_)
L2=ones(1,800)';
for i=1:100
    L2(end+1)=2;
end
for i=1:100
    L2(end+1)=3;
end

% ������ �з��� ���� ���
e1=error_calculate(Y1,L1);
e2=error_calculate(Y2,L2);

disp(['X5�� ���� ������ �з� ����:']);
disp(e1);
disp(['X5_�� ���� ������ �з� ����:']);
disp(e2);

% ��Ŭ����� �з��� ���� ���
e3=error_calculate(Y3,L1);
e4=error_calculate(Y4,L2);

disp(['X5�� ���� ��Ŭ����� �з� ����:']);
disp(e3);
disp(['X5_�� ���� ��Ŭ����� �з� ����:']);
disp(e4);
