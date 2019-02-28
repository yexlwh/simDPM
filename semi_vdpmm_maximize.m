function [params kldiv] = semi_vdpmm_maximize(testData,params,gammas,label,gammas_1)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
%priors
data=testData;
[N D] = size(data);
K = numel(params.a);
a0 = D;
beta0 = 1;
mean0 = mean(data);
B0 = .1 * D * cov(data);

%convenience variables first
Ns = sum(gammas,1) + 1e-10;
mus = zeros(K,D);
sigs = zeros(D,D,K);
mus = gammas' * data ./ repmat(Ns',1,D);
%ag2 = shiftdim(repmat(sqrt(gammas),1,1,D),2);
for i = 1:K
    %            mus(i,:) = sum(repmat(gammas(:,i),1,D).*data) / Ns(i);
    diff0 = data - repmat(mus(i,:),size(data,1),1);
    diff1 = repmat(sqrt(gammas(:,i)),1,D) .* diff0;
    %diff2 = ag2(:,:,i)' .* diff0;
    sigs(:,:,i) = diff1' * diff1;
end

%now the estimates for the variational parameters
params.g(:,1) = 1 + sum(gammas);
%g_{s,2} = Eq[alpha] + sum_n sum_{j=s+1} gamma_j^n
params.g(:,2) = params.eq_alpha + ...
    flipdim(cumsum(flipdim(sum(gammas),2)),2) - sum(gammas);
params.beta = Ns + beta0;
params.a = Ns + a0;
params.mean = (repmat(Ns',1,D) .* mus + beta0 * repmat(mean0,size(Ns,2),1)) ./ repmat(Ns' + beta0,1,D);
for i = 1:K
    diff = mus(i,:) - mean0;
    params.B(:,:,i) = sigs(:,:,i) + Ns(i) * beta0 * diff * diff' / (Ns(i)+beta0) + B0;
end

%%
% %update the semi parameters
% uniLabel=unique(label);
% tempSigma=zeros(D,D,K);
% tempMean=zeros(D,K);
% for k=1:K
%     flag=find(k==uniLabel);
%     if ~isempty(flag)
%         tempMean(:,k)=params.meanN(k,:)';
%         tempSigma(:,:,k)=params.sigma(:,:,k);
%         continue;
%     end
%     flag=[];
%     
% %     for i=1:N
% %         tempMean(:,k)=gammas(i,k)*data(i,:)'+tempMean(:,k);
% %     end;
% %     tempMean(:,k)=(1/sum(gammas(:,k)))*tempMean(:,k);
%     tempMean(:,k)=(gammas_1(:,k)'*data)';
%     tempMean(:,k)=(1/sum(gammas_1(:,k)))*tempMean(:,k);
%     %tempMean(:,k)=(1/N)*tempMean(:,k);
% %     for i=1:N
% %         tempSigma(:,:,k)=tempSigma(:,:,k)+gammas(i,k)*(data(i,:)'-tempMean(:,k))*(data(i,:)-tempMean(:,k)');
% %     end;
%     diff0=data-repmat(tempMean(:,k)',N,1);
%     diff1 = repmat(sqrt(gammas_1(:,k)),1,D) .* diff0;
%     tempSigma(:,:,k)=diff1' * diff1;
%     
%     tempSigma(:,:,k)=(1/sum(gammas_1(:,k)))*tempSigma(:,:,k);
%     selDet=det(tempSigma(:,:,k));
% %     if selDet<1.0e-15
% %         tempSigma(:,:,k)=params.sigma(:,:,k);
% %         tempMean(:,k)= params.mean(:,k);
% %     else
%      sigmaDet=det(tempSigma(:,:,k));
%      if sigmaDet<1.0e-10
%         [sN, sD]=size(tempSigma(:,:,k));
%         tempSigma(:,:,k)=tempSigma(:,:,k)+1.0e-5*eye(sD);
%      end;
% end;
% params.meanN=tempMean';
% params.sigma=tempSigma;

kldiv=0;

end

