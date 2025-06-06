%%数据导入
%预处理
clc
clear
close all;
load D:\dpanzhuomian\XY.txt
X=XY
load D:\dpanzhuomian\Y.txt
%波段筛选
load D:\dpanzhuomian\SNV结果及图像\特征波段筛选\vcpa\X.txt
X_transposed = transpose(X);
X=X_transposed
%%
clc
clear
close all;
load D:\dpanzhuomian\X.txt
load D:\dpanzhuomian\Y.txt
%%  SNV预处理
Xnir = X(1:end,1:end) ;
[me, ] = mean(Xnir);
[m, n] = size(Xnir);
Xm = mean(Xnir, 2);
dX = Xnir - repmat(Xm, 1, n);
Xsnv = dX./ repmat(sqrt(sum(dX.^2,2)/n), 1, n) ;
X = Xsnv;
%% MSC预处理
Xnir = X(1:end,1:end);
[me,] = mean(Xnir);
[m, n] = size(Xnir);
for i = 1:m    
  p = polyfit(me, Xnir(i,:),1);    
  Xmsc(i,:) = (Xnir(i,:) - p(2) * ones(1, n))./(p(1) * ones(1, n));
end
X = Xmsc;
%%  S-G预处理（可能不太好用a）
[x_sg]=savgol(X,15,2,0);%一阶求导
X = x_sg;
%% 一维中值滤波medfilt预处理
X=medfilt1(X);
%% z-score（融合yongde）
X = zscore(X);
%%  CARS特征波段提取
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
%%  BOSS特征波段提取(（需要自己找波段到excel里，再导入进来））
Xcal = X;
ycal = Y;
nLV_max=10;
fold=5;
method='center';
num_bootstrap=1000;
BOSS=boss(Xcal,ycal,nLV_max,fold,method,num_bootstrap,0);
%%  最小主成分(（特征波段提取后是几,X后就改成几,看X-test)）
x=[1:23];%特征波段提取后的特征数,改下边的六个数值-针对PLSR
Model=ipls(X,Y,20,'mean',1,x,'syst123',5);%第一个数是想看8个柱子，值就是8；
plsrmse(Model,0);
%% 
% 中心化
X_mean = mean(X);
X_center = X - X_mean;

% 计算协方差矩阵
cov_mat = cov(X_center);

% 特征值分解
[V,D] = eig(cov_mat);
eigenvalues = diag(D);
[~,idx] = sort(eigenvalues,'descend');
V_sort = V(:,idx);

% 选择前k个特征向量
k = 8;
V_k = V_sort(:,1:k);

% 投影到新的坐标系中
X_pca = X_center * V_k;

% 可视化降维后的数据
figure;
scatter(X_pca(:,1),X_pca(:,2),15,Y,'filled');
xlabel('PC1');
ylabel('PC2');
title('PCA');
X = X_pca;
%% 111111PLSR建模
%%% Step four - Partial least squares (PLS) model building
[p_train,p_test,t_train,t_test]=ks(X,Y,112);%样品集合划分3：1，135为3
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
line([0,20],[0,20],'color','b','LineStyle','-') % TRS画出Y=X的这条线
legend('Rc=0.968 ','Rp=0.976','R^2=1');
%%  22222PLSR划分训练集验证集
[Xtrain,Xtest,Ytrain,Ytest]=ks(X,Y,112)
X = Xtrain;
Y = Ytrain;
Xpred = Xtest;
Yref = Ytest;
% CARS-PLS方法
x=[1:7];%光谱区间设定
%利用iPLS函数对训练集建立全局偏最小二乘模型
Model=ipls(X,Y,7,'mean',1,x,'syst123',5);%20个区间改为1个区间，就是PLS方法（训练模型建立）如果不是光谱，嗅觉可视化的数据就将xsnv变为xcal，ytr变为ycal
plsrmse(Model,0);%得出主成分数，0是个参数；通过此函数画出利用不同主城分数所建模型得RMSECV，取最小误差主成分
% plspvsm(Model,10,1);%1个区间的5个主成分画图，是上一步得出的 因为是全局变量 所以只有1个区间
%通过plsmodel建立样本预测模型
oneModel=plsmodel(Model,1,7,'mean','test',5);%（预测模型建立，如果不是光谱，嗅觉可视化的数据就将xtnv变为xtest，yte变为ytest)(还是用建立的模型进行预测）
predModel=plspredict(Xpred,oneModel,7,Yref);
%plspvsm(predModel,10);%预测集的结果
Yptr=Model.PLSmodel{1, 1}.Ypred(:,:,7); %提取训练集预测值
Ypt=predModel.Ypred(:,:,7);%提取预测集预测值
RC=(min(min(corrcoef(Yptr,Ytrain))))%训练集相关系数
RP=(min(min(corrcoef(Ypt,Ytest))))%预测集相关系数
clear mse
RMSEC=sqrt(mse(Yptr-Ytrain))
RMSEP=sqrt(mse(Ypt,Ytest))
RSDP=100*std(Ypt)./mean(Ypt)
RPDp=std(Ypt)./RMSEP
T=[RC,RMSEC,RP,RMSEP,RSDP,RPDp]%剩余预测残差
%%  Ntree 1设置的树的数目 2划分训练集验证集
[Xtrain,Xtest,Ytrain,Ytest]=ks(X,Y,490)
% 优选最优树
TTRSMEC=[];
for i=25:25:1000
    TR=[];   
       for j=1:1:10
xtr = Xtrain;
ytr = Ytrain;
xte = Xtest;
yte = Ytest;
model=regRF_train(xtr,ytr,i,1);%%   ·
Yptr =regRF_predict(xtr,model);
error=ytr-Yptr;
RMSEC=sqrt(mse(error));%min(min(corrcoef(error)))   %sqrt(mse(error))
TR=[TR;RMSEC];
    end
  TTRSMEC=[TTRSMEC TR];   
end
%%
first_row = TTRSMEC(1, :); % 取出第一行
[min_value, col_index] = min(first_row); % 取出最小值和对应的列数
Ttree = col_index*25; % RSMEC最小的树
for i=Ttree %树
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

RC=min(min(corrcoef(Yptr,ytr)));%相关系数
RP=min(min(corrcoef(Ypt,yte)));
RMSEC=sqrt(mse(Yptr-ytr));%均方根误差
RMSEP=sqrt(mse(Ypt-yte));
CVc=std(Yptr)./mean(Yptr);%变异系数
CV=std(Ypt)./mean(Ypt) ; %变异系数
RPDc=std(Yptr)./RMSEC; %剩余预测残差 
RPDp=std(Ypt)./RMSEP;
SEC=std(Yptr,ytr)/sqrt(95);%标准误差 
SEP=std(Ypt,yte)/sqrt(43);%标准误差
Biasp=mean(Ypt)-mean(yte);%偏差
Biasc=mean(Yptr)-mean(ytr);%偏差
VC=var(Yptr);%方差
VP=var(Ypt);%方差
aa=[RC RMSEC Biasc]                     %训练性能指标集合
aa2=[RP RMSEP Biasp SEP CV RPDp ]            %预测集性能指标集合
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

 %%  %% SVR  划分训练集验证集
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
% S.E.=Std(Ypt) /Sqrt(45) %%均值标准误差
T=[RC,RMSEC,RP,RMSEP,RSDP,RPDp]%剩余预测残差
%%
plot(t_train,Yptr,'ro'); %：画图 预测集样本测定值与预测值散点图
hold on
plot(t_test,Ypt,'b*');
% set(gcf,'paperunits','centimeters');
% set(gcf,'papersize',[17 5]);
line([0,20],[0,20],'color','b','LineStyle','-') % TRS画出Y=X的这条线
%legend('Rc=0.9156 ,RMSEC=0.22 ','Rp=0.8628, RMSEP=0.41');
legend('R2c=0.9863 ','R2p=0.9541','R^2=1');
%%
xlabel('Calibation values','FontName','Times New Roman','FontSize',25);%设置了x轴的标签为“”，并指定了字体为“Times New Roman”，字体大小为25。
ylabel('Prediction values','FontName','Times New Roman','FontSize',25);
xlim([1.5,4.5]);%设置坐标轴范围1.5到4.5
ylim([1.5,4.5]);
line('LineStyle','none')
line([1.5,4.5],[1.5,4.5],'color','b','LineStyle','-') %绘制了一条从(1.5,1.5)到(4.5,4.5)的蓝色直线（'color','b'和'LineStyle','-'）
h = legend('Rc=0.99 ,RMSEC=0.09 ','Rp=0.94, RMSEP=0.22','RPD=2.824','FontName','Times New Roman','FontSize',20);%模型的性能指标。图例的字体设置为“Times New Roman”，字体大小为20。
set(h, 'Box','off');%将图例的边框设置为不可见（'Box','off'）