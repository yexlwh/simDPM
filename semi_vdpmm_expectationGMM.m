function [params gammas loglike] = semi_vdpmm_expectationGMM(testData,params,infinite,gammas,label,model,s,W,cho)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
data=testData;
[N D]=size(data);
K = size(params.a,2);
eq_log_Vs = zeros(K,1);
eq_log_1_Vs = zeros(K,1);
log_gamma_tilde = zeros(size(data,1),K);
uniqueLabel=unique(label);
% for i=1:size(uniqueLabel,2)
%     params.g(uniqueLabel(i),1)=params.g1(uniqueLabel(i),1);
%     params.g(uniqueLabel(i),2)=params.g1(uniqueLabel(i),2);
% end;
modelN=size(model,2);
for i=1:modelN
    Px = calc_probGaussian(model(i),testData); 
    pPi=model(i).Pi;
    L(:,i) =( log(Px*pPi'+1e-10));
end;

j=1;
for i = 1:K
    eq_log_Vs(i) = digamma(params.g(i,1)) - digamma(params.g(i,1)+params.g(i,2));
    eq_log_1_Vs(i) = digamma(params.g(i,2)) - digamma(params.g(i,1)+params.g(i,2));
    log_V_prob(i) = eq_log_Vs(i) + sum(eq_log_1_Vs(1:(i-1)));
    
    eq_log_pi(i) = digamma(params.g(i,1)) - digamma(sum(params.g(:,1)));
   
    flag=find(label==i);
    %if isempty(flag)
        pob(:,i) = normwish(data,params.mean(i,:),params.beta(i),params.a(i),params.B(:,:,i));
    %else
        %pob(:,i) = (log(GaussPDF(data',params.meanN(i,:)',params.sigma(:,:,i))))';
        %pob(:,i) = L(:,i);
        j=j+1;
    %end;
    %pob_1(:,i) = (log(GaussPDF(data',params.meanN(i,:)',params.sigma(:,:,i))))';
    %pob(:,i)=pob_1(:,i);
    %disp(num2str(max(abs(pob - data_specific - normalizer))));
    if (infinite)
        log_gamma_tilde(:,i) = log_V_prob(i) + repmat(s(i),N,1).*pob(:,i)+repmat(1-s(i),N,1).*L(:,i);
        %log_gamma_tilde_1(:,i) = log_V_prob(i) + pob_1(:,i);
    else
        log_gamma_tilde(:,i) = eq_log_pi(i) + repmat(s(i),N,1).*pob(:,i)+repmat(1-s(i),N,1).*L(:,i);
    end
end

%this should involve some KL-divergences

%%
%gammas_1 = exp(log_gamma_tilde_1);
%gammas_1 = gammas_1 ./ repmat(sum(gammas_1,2),1,K);
%params.ll = sum(sum(gammas .* log_gamma_tilde));

%use the laplace regularier to capture manifold struture.
lambdaRate=1000;
% if cho==1
%     count=100;
%     while count~=0
%         if count==50
%             lambdaRate=0.02;
%         end;
%         if count==90
%             lambdaRate=0.01;
%         end;
%         gammas1=(1-lambdaRate).*gammas;
%         W1=repmat(sum(W,2),1,size(gammas,2));
%         gammas2=lambdaRate.*(W*gammas);
%         gammas=gammas1+(gammas2)./W1;
%         if mod(count,10)==0
%             lambdaRate=0.9*lambdaRate;
%         end;
%         count=count-1;
%     end;
% end;
if cho==1
    gammas=inv(lambdaRate.*eye(N)-W)*(lambdaRate.*gammas);
end;
gammas = exp(log_gamma_tilde);
gammas = gammas ./ repmat(sum(gammas,2),1,K);
%params.ll = sum(sum(gammas .* log_gamma_tilde));
loglike = sum(sum(gammas .* pob));
%reestimating eq_alpha
h1 = 1; h2 = 1;
w1 = h1 + K - 1;
w2 = h2 - sum(eq_log_1_Vs(1:(end-1)));
params.eq_alpha = w1 / w2;


end

