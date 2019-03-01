clear all;
inputData=load('1R2RCR_truth.mat')
x=inputData.x;
s=inputData.s;
[C N F]=size(x);
for i=1 : N
    data(i,:)=reshape(x(:,i,:),1,C*F);
end

[engVector engValue]=PCA(data);
xPca=data*engVector(:,1:10);

c=s*10;
figure(1)
xPca=xPca*100;
scatter3(xPca(:,1),xPca(:,2),xPca(:,3),c,s)

fea=xPca;
label=s;



%%
%select train data
supCluster=[];
train=[];
trainLabel=[];

testAll=fea;

%construt nearest neighbor graph
options.show = 0;
options.lambda = 0;
options.k = 5;
W = constructW(testAll, options);

% [params gammas assign,initGammas]=semiDpmmGMM(train,[],trainLabel,testAll);
[params gammas assign initGammas]=DpmmGMMManifold(train,[],trainLabel,testAll,W);
% K=size(unique(label),1);
% [kc,Kcen]=kmeans(testAll,K);
% assign=kc';
subplot(1,3,1);
scatterMixture(testAll,label);
subplot(1,3,2);
scatterMixture(testAll,assign);
[cen,Kl]=kmeans(testAll,3);
subplot(1,3,3);
scatterMixture(testAll,cen);

labelAll=label;
assign=assign';
assignC=cen;
