function [params gammas assign initGammas] = DpmmGMMManifold(trainData, varargin, label, testData,W)

%[K infinite verbose maxits minits eps] = ...
%    process_options(varargin,'K',50,'infinite',1,'verbose',1,...
%    'maxits',500,'minits',10,'eps',.01);
K=30;
infinite=1;
verbose=1;
maxits=40;
minits=10;
eps=0.01;
[N D]=size(testData);

[params gammas] = vdpmm_init(testData,K);
initGammas=gammas;
gammas_1=initGammas;

%[params]=sdpTrain_1(trainData,label,K);
Kseeds=[2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,5,5,5,5,5,5,5];
% uniqueLabel=sort(unique(label));
% indexTrain=find(label==uniqueLabel(1));
[modelTemp,labelpx]=gmm(testData,2);
for count=1:K
    model(count)=modelTemp;
end;
% for count=1:size(uniqueLabel,2)
%     indexTrain=find(label==uniqueLabel(count));
%     [modelTemp,labelpx]=gmm(trainData(indexTrain,:),Kseeds(count));
%     model(uniqueLabel(count))=modelTemp;
% end;

numits = 2;
score = -inf;
score_change = inf;
for count=1:K
%     flag=find(uniqueLabel==count);
%     if isempty(flag)
        s(1,count)=1;%不在标签内
%     else
%         s(1,count)=0;
%     end;
end;
sT=s;
H=1000;
while (numits < maxits) %&& (numits < minits || score_change > 1e-4)
    if numits>13
        bre=0;
    end;
    score_change = score;
    [params(numits) kldiv] = semi_vdpmm_maximize(testData,params(numits-1),gammas,label,gammas_1);
    [params(numits) gammas loglike] = semi_vdpmm_expectationGMM(testData,params(numits),infinite,gammas,label,model,s,W,1);
    [model] = gmmMSTEP(testData, 2,model,H,gammas,label);
%     prev_ll = params(numits).ll;
%repmat(1-sT,N,1).*
     numits = numits+1;
%     score = kldiv + loglike;
%     score_change = (score - score_change) / abs(score);
     if (verbose), disp(sprintf('Iteration: %i\tscore: %f\tdelta: %f',numits,score,score_change)); end
end
uniqueLabel=unique(label);
% for i=1:size(uniqueLabel,2)
%     gammas(:,uniqueLabel(i))=gammas_1(:,uniqueLabel(i));
% end;
[ig assign] = max(gammas');   


end