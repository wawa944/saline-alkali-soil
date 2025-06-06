%%���ݵ���
%Ԥ����
clc
clear
close all;
load D:\dpanzhuomian\XY.txt
X=XY
load D:\dpanzhuomian\Y.txt
%����ɸѡ
load D:\dpanzhuomian\SNV�����ͼ��\��������ɸѡ\vcpa\X.txt
X_transposed = transpose(X);
X=X_transposed
%%
clc
clear
close all;
load D:\dpanzhuomian\X.txt
load D:\dpanzhuomian\Y.txt
%%  SNVԤ����
Xnir = X(1:end,1:end) ;
[me, ] = mean(Xnir);
[m, n] = size(Xnir);
Xm = mean(Xnir, 2);
dX = Xnir - repmat(Xm, 1, n);
Xsnv = dX./ repmat(sqrt(sum(dX.^2,2)/n), 1, n) ;
X = Xsnv;
%% MSCԤ����
Xnir = X(1:end,1:end);
[me,] = mean(Xnir);
[m, n] = size(Xnir);
for i = 1:m    
  p = polyfit(me, Xnir(i,:),1);    
  Xmsc(i,:) = (Xnir(i,:) - p(2) * ones(1, n))./(p(1) * ones(1, n));
end
X = Xmsc;
%%  S-GԤ�������ܲ�̫����a��
[x_sg]=savgol(X,15,2,0);%һ����
X = x_sg;
%% һά��ֵ�˲�medfiltԤ����
X=medfilt1(X);
%% z-score���ں�yongde��
X = zscore(X);
%%  CARS����������ȡ
A=10; % A can be set to a large number.
K=5;  % fold number of CV for model validation
N=60; %+++ number of evolution
y = Y;
CARS=carspls(X,y,A,K,'center',N); 

SJ = CARS.retained_variables
[a, b]=size(X)
for i = 1:size(X, 1)
    for j = SJ(:)
        x = X(:,j)
    end
end
X = x;
%%  BOSS����������ȡ(����Ҫ�Լ��Ҳ��ε�excel��ٵ����������
Xcal = X;
ycal = Y;
nLV_max=10;
fold=5;
method='center';
num_bootstrap=1000;
BOSS=boss(Xcal,ycal,nLV_max,fold,method,num_bootstrap,0);
%%  ��С���ɷ�(������������ȡ���Ǽ�,X��͸ĳɼ�,��X-test)��
x=[1:23];%����������ȡ���������,���±ߵ�������ֵ-���PLSR
Model=ipls(X,Y,20,'mean',1,x,'syst123',5);%��һ�������뿴8�����ӣ�ֵ����8��
plsrmse(Model,0);
%% 
% ���Ļ�
X_mean = mean(X);
X_center = X - X_mean;

% ����Э�������
cov_mat = cov(X_center);

% ����ֵ�ֽ�
[V,D] = eig(cov_mat);
eigenvalues = diag(D);
[~,idx] = sort(eigenvalues,'descend');
V_sort = V(:,idx);

% ѡ��ǰk����������
k = 8;
V_k = V_sort(:,1:k);

% ͶӰ���µ�����ϵ��
X_pca = X_center * V_k;

% ���ӻ���ά�������
figure;
scatter(X_pca(:,1),X_pca(:,2),15,Y,'filled');
xlabel('PC1');
ylabel('PC2');
title('PCA');
X = X_pca;
%% 111111PLSR��ģ
%%% Step four - Partial least squares (PLS) model building
[p_train,p_test,t_train,t_test]=ks(X,Y,112);%��Ʒ���ϻ���3��1��135Ϊ3
Xtrain=p_train
Ytrain=t_train
Xtest=p_test
Ytest=t_test
A_max=18; % A_max: the maximal principle component to extract.
fold=10;% fold: the group number for cross validation.
method='center';% method: data pretreatment method, contains: autoscaling, pareto,minmax,center or none.
CV=plscvfold(Xtrain,Ytrain,A_max,fold,method);% cross validation of PLS to select the best PLS component
A=CV.optPC; % The best PLS component 
PLS=pls(Xtrain,Ytrain,A,method);
Xtest_expand=[Xtest ones(size(Xtest,1),1)];
coef=PLS.coef_origin;
ypred=Xtest_expand*coef(:,end);

% Model assessment
SST=sum((Ytest-mean(Ytest)).^2); 
SSE=sum((Ytest-ypred).^2); 
R2_C=PLS.R2; 
R2_P=1-SSE/SST;
RMSEC=sqrt(PLS.SSE/size(Xtrain,1));
RMSEP=sqrt(SSE/size(Xtest,1));
RPDp=std(Ytest)./RMSEP

% plot the correlation diagrams between the predicted values and the reference values 
ypred_test = [ypred,Ytest];
plot(Ytrain,PLS.y_est,'*r')
hold on
plot(Ytest,ypred,'ob') % plot the figure
line([0,20],[0,20],'color','b','LineStyle','-') % TRS����Y=X��������
legend('Rc=0.968 ','Rp=0.976','R^2=1');
%%  22222PLSR����ѵ������֤��
[Xtrain,Xtest,Ytrain,Ytest]=ks(X,Y,112)
X = Xtrain;
Y = Ytrain;
Xpred = Xtest;
Yref = Ytest;
% CARS-PLS����
x=[1:7];%���������趨
%����iPLS������ѵ��������ȫ��ƫ��С����ģ��
Model=ipls(X,Y,7,'mean',1,x,'syst123',5);%20�������Ϊ1�����䣬����PLS������ѵ��ģ�ͽ�����������ǹ��ף�������ӻ������ݾͽ�xsnv��Ϊxcal��ytr��Ϊycal
plsrmse(Model,0);%�ó����ɷ�����0�Ǹ�������ͨ���˺����������ò�ͬ���Ƿ�������ģ�͵�RMSECV��ȡ��С������ɷ�
% plspvsm(Model,10,1);%1�������5�����ɷֻ�ͼ������һ���ó��� ��Ϊ��ȫ�ֱ��� ����ֻ��1������
%ͨ��plsmodel��������Ԥ��ģ��
oneModel=plsmodel(Model,1,7,'mean','test',5);%��Ԥ��ģ�ͽ�����������ǹ��ף�������ӻ������ݾͽ�xtnv��Ϊxtest��yte��Ϊytest)(�����ý�����ģ�ͽ���Ԥ�⣩
predModel=plspredict(Xpred,oneModel,7,Yref);
%plspvsm(predModel,10);%Ԥ�⼯�Ľ��
Yptr=Model.PLSmodel{1, 1}.Ypred(:,:,7); %��ȡѵ����Ԥ��ֵ
Ypt=predModel.Ypred(:,:,7);%��ȡԤ�⼯Ԥ��ֵ
RC=(min(min(corrcoef(Yptr,Ytrain))))%ѵ�������ϵ��
RP=(min(min(corrcoef(Ypt,Ytest))))%Ԥ�⼯���ϵ��
clear mse
RMSEC=sqrt(mse(Yptr-Ytrain))
RMSEP=sqrt(mse(Ypt,Ytest))
RSDP=100*std(Ypt)./mean(Ypt)
RPDp=std(Ypt)./RMSEP
T=[RC,RMSEC,RP,RMSEP,RSDP,RPDp]%ʣ��Ԥ��в�
%%  Ntree 1���õ�������Ŀ 2����ѵ������֤��
[Xtrain,Xtest,Ytrain,Ytest]=ks(X,Y,490)
% ��ѡ������
TTRSMEC=[];
for i=25:25:1000
    TR=[];   
       for j=1:1:10
xtr = Xtrain;
ytr = Ytrain;
xte = Xtest;
yte = Ytest;
model=regRF_train(xtr,ytr,i,1);%%   ��
Yptr =regRF_predict(xtr,model);
error=ytr-Yptr;
RMSEC=sqrt(mse(error));%min(min(corrcoef(error)))   %sqrt(mse(error))
TR=[TR;RMSEC];
    end
  TTRSMEC=[TTRSMEC TR];   
end
%%
first_row = TTRSMEC(1, :); % ȡ����һ��
[min_value, col_index] = min(first_row); % ȡ����Сֵ�Ͷ�Ӧ������
Ttree = col_index*25; % RSMEC��С����
for i=Ttree %��
          for j=3 %pcs
xtr = Xtrain;
ytr = Ytrain;
xte = Xtest;
yte = Ytest;
model=regRF_train(xtr,ytr,i,1);
Yptr =regRF_predict(xtr,model);
Ypt =regRF_predict(xte,model);
 end
end

RC=min(min(corrcoef(Yptr,ytr)));%���ϵ��
RP=min(min(corrcoef(Ypt,yte)));
RMSEC=sqrt(mse(Yptr-ytr));%���������
RMSEP=sqrt(mse(Ypt-yte));
CVc=std(Yptr)./mean(Yptr);%����ϵ��
CV=std(Ypt)./mean(Ypt) ; %����ϵ��
RPDc=std(Yptr)./RMSEC; %ʣ��Ԥ��в� 
RPDp=std(Ypt)./RMSEP;
SEC=std(Yptr,ytr)/sqrt(95);%��׼��� 
SEP=std(Ypt,yte)/sqrt(43);%��׼���
Biasp=mean(Ypt)-mean(yte);%ƫ��
Biasc=mean(Yptr)-mean(ytr);%ƫ��
VC=var(Yptr);%����
VP=var(Ypt);%����
aa=[RC RMSEC Biasc]                     %ѵ������ָ�꼯��
aa2=[RP RMSEP Biasp SEP CV RPDp ]            %Ԥ�⼯����ָ�꼯��
%%
plot(ytr,Yptr,'c.'); 
hold on
plot(yte,Ypt,'md');
% set(gcf,'paperunits','centimeters');
% set(gcf,'papersize',[17 5]);
xlabel('Calibation values','FontName','Times New Roman','FontSize',25);
ylabel('Prediction values','FontName','Times New Roman','FontSize',25);
xlim([0,450]);
ylim([0,450]);
line('LineStyle','none')
line([0,450],[0,450],'color','b','LineStyle','-') 
h = legend('Rc=0.98 ,RMSEC=13.72 ','Rp=0.97, RMSEP=21.57','RPD=5.935','FontName','Times New Roman','FontSize',20);
set(h, 'Box','off');

 %%  %% SVR  ����ѵ������֤��
[p_train,p_test,t_train,t_test]=ks(X,Y,112)
tic
xtr=p_train;
ytr=t_train;
xte=p_test;
yte=t_test;
train_y = ytr;
train_x =xtr;
test_y = yte;
test_x = xte;
Method_option.plotOriginal = 0;
Method_option.xscale = 1;
Method_option.yscale =1;
Method_option.plotScale = 0;
Method_option.pca =1;
Method_option.type =1;
[predict_Y,mse,r] = SVR(train_y,train_x,test_y,test_x,Method_option);
ycal=Y
Xcal=X
%[predict_YT,mse,r] = SVR(train_y,train_x,train_y,train_x,Method_option);
cpu_time=toc
Yptr=predict_Y{1};
Ypt=predict_Y{2};
RC=(min(min(corrcoef(Yptr,t_train))))
RP=(min(min(corrcoef(Ypt,t_test))))
clear mse
RMSEC=sqrt(mse(Yptr,t_train))
RMSEP=sqrt(mse(Ypt,t_test))
RSDP=100*std(t_test)./mean(Ypt)
RPDp=std(t_test)./RMSEP
% S.E.=Std(Ypt) /Sqrt(45) %%��ֵ��׼���
T=[RC,RMSEC,RP,RMSEP,RSDP,RPDp]%ʣ��Ԥ��в�
%%
plot(t_train,Yptr,'ro'); %����ͼ Ԥ�⼯�����ⶨֵ��Ԥ��ֵɢ��ͼ
hold on
plot(t_test,Ypt,'b*');
% set(gcf,'paperunits','centimeters');
% set(gcf,'papersize',[17 5]);
line([0,20],[0,20],'color','b','LineStyle','-') % TRS����Y=X��������
%legend('Rc=0.9156 ,RMSEC=0.22 ','Rp=0.8628, RMSEP=0.41');
legend('R2c=0.9863 ','R2p=0.9541','R^2=1');
%%
xlabel('Calibation values','FontName','Times New Roman','FontSize',25);%������x��ı�ǩΪ��������ָ��������Ϊ��Times New Roman���������СΪ25��
ylabel('Prediction values','FontName','Times New Roman','FontSize',25);
xlim([1.5,4.5]);%���������᷶Χ1.5��4.5
ylim([1.5,4.5]);
line('LineStyle','none')
line([1.5,4.5],[1.5,4.5],'color','b','LineStyle','-') %������һ����(1.5,1.5)��(4.5,4.5)����ɫֱ�ߣ�'color','b'��'LineStyle','-'��
h = legend('Rc=0.99 ,RMSEC=0.09 ','Rp=0.94, RMSEP=0.22','RPD=2.824','FontName','Times New Roman','FontSize',20);%ģ�͵�����ָ�ꡣͼ������������Ϊ��Times New Roman���������СΪ20��
set(h, 'Box','off');%��ͼ���ı߿�����Ϊ���ɼ���'Box','off'��