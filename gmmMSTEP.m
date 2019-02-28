function [model] = gmmMSTEP(X, K_or_centroids,model,H,Phi,trainLabel)  
% ============================================================  
% Expectation-Maximization iteration implementation of  
% Gaussian Mixture Model.  
%  
% PX = GMM(X, K_OR_CENTROIDS)  
% [PX MODEL] = GMM(X, K_OR_CENTROIDS)  
%  
%  - X: N-by-D data matrix.  
%  - K_OR_CENTROIDS: either K indicating the number of  
%       components or a K-by-D matrix indicating the  
%       choosing of the initial K centroids.  
%  
%  - PX: N-by-K matrix indicating the probability of each  
%       component generating each point.  
%  - MODEL: a structure containing the parameters for a GMM:  
%       MODEL.Miu: a K-by-D matrix.  
%       MODEL.Sigma: a D-by-D-by-K matrix.  
%       MODEL.Pi: a 1-by-K vector.  
% ============================================================  
% @SourceCode Author: Pluskid (http://blog.pluskid.org)  
% @Appended by : Sophia_qing (http://blog.csdn.net/abcjennifer)  
      
  
%% Generate Initial Centroids  
    threshold = 1e-15;  
    [N, D] = size(X);  
   
    if isscalar(K_or_centroids) %if K_or_centroid is a 1*1 number  
        K = K_or_centroids;  
        Rn_index = randperm(N); %random index N samples  
        centroids = X(Rn_index(1:K), :); %generate K random centroid  
    else % K_or_centroid is a initial K centroid  
        K = size(K_or_centroids, 1);   
        centroids = K_or_centroids;  
    end  
   
    uniqueLabel=unique(trainLabel);
    Knum=size(Phi,2);
    for uk=1:Knum
        PhiK=repmat(Phi(:,uk),1,K);
        flag=find(uniqueLabel==uk);
        if ~isempty(flag)
            continue;
        end;
        %% initial values  
        [pMiu pPi pSigma] = init_params();  

        Lprev = -inf; %��һ�ξ�������  

        %% EM Algorithm  
        epsilon=1e-10.*ones(N,K);
        while true  
            %% Estimation Step  
            Px = calc_prob();  

            % new value for pGamma(N*k), pGamma(i,k) = Xi�ɵ�k��Gaussian���ɵĸ���  
            % ����˵xi����pGamma(i,k)���ɵ�k��Gaussian���ɵ�  
            pGamma = Px .* repmat(pPi, N, 1)+epsilon; %���� = pi(k) * N(xi | pMiu(k), pSigma(k))  
            pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K); %��ĸ = pi(j) * N(xi | pMiu(j), pSigma(j))������j���  

            %% Maximization Step - through Maximize likelihood Estimation  

            Nk = sum(pGamma, 1); %Nk(1*k) = ��k����˹����ÿ�������ĸ��ʵĺͣ�����Nk���ܺ�ΪN��  
            pNk=sum(pGamma.*PhiK, 1);
            % update pMiu  
            
%             Xtemp=circshift(X,1);
%             Xtemp(1,:)=zeros(1,D);
            pMiu(1,:)=1./(pNk(1)+H).*( (pGamma(:,1).*PhiK(:,1))'* X);
            for count=2:K
                pMiu(count,:)=1./(pNk(count)+H).*( (pGamma(:,count).*PhiK(:,count))'* X+H.*pMiu(count-1,:));
                %pMiu = diag(1./(pNk+H)) *( (pGamma.*PhiK)'* X+H*Xtemp); %update pMiu through MLE(ͨ����� = 0�õ�)  
            end;
            pPi = Nk/N;  
            % update k�� pSigma  
            for kk = 1:K
                if kk==1
                    Xshift = X-repmat(pMiu(kk, :), N, 1);  
                    pSigma(:, :, kk) = (Xshift' * ...  
                        (diag(pGamma(:, kk).*PhiK(:,kk)) * Xshift)) / (pNk(kk)+10);  
                else
                    Xshift = X-repmat(pMiu(kk, :), N, 1);  
                    pSigma(:, :, kk) = (Xshift' * ...  
                        (diag(pGamma(:, kk).*PhiK(:,kk)) * Xshift)) / (pNk(kk)+10);%-H*(pMiu(kk, :)-pMiu(kk-1, :))'*(pMiu(kk, :)-pMiu(kk-1, :));  
                end;
            end  

            % check for convergence  
            L = sum(log(Px*pPi'));  
            if L-Lprev < threshold  
                break;  
            end  
            Lprev = L;  
        end  

    %     if nargout == 1  
    %         varargout = {Px};  
    %     else  
    %         model = [];  
    %         model.Miu = pMiu;  
    %         model.Sigma = pSigma;  
    %         model.Pi = pPi;  
    %         varargout = {Px, model};  
    %     end  
          modelTemp = [];  
          modelTemp.Miu = pMiu;  
          modelTemp.Sigma = pSigma;  
          modelTemp.Pi = pPi;  
          model(uk)=modelTemp;
    end;
   
    %% Function Definition  
      
    function [pMiu pPi pSigma] = init_params()  
        pMiu = centroids; %k*D, ��k������ĵ�  
        pPi = zeros(1, K); %k��GMM��ռȨ�أ�influence factor��  
        pSigma = zeros(D, D, K); %k��GMM��Э�������ÿ����D*D��  
   
        % ������󣬼���N*K�ľ���x-pMiu��^2 = x^2+pMiu^2-2*x*Miu  
        distmat = repmat(sum(X.*X, 2), 1, K) + ... %x^2, N*1�ľ���replicateK��  
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - ...%pMiu^2��1*K�ľ���replicateN��  
            2*X*pMiu';  
        [~, labels] = min(distmat, [], 2);%Return the minimum from each row  
   
        for k=1:K  
            Xk = X(labels == k, :);  
            pPi(k) = size(Xk, 1)/N;  
            pSigma(:, :, k) = cov(Xk);  
        end  
    end  
   
    function Px = calc_prob()   
        %Gaussian posterior probability   
        %N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))  
        Px = zeros(N, K);  
        for k = 1:K  
            Xshift = X-repmat(pMiu(k, :), N, 1); %X-pMiu 
            if det(pSigma(:, :, k))<1e-6
                Px(:, k)=1e-8/N;
                continue;
            end;
            
%             try
            
            inv_pSigma = inv(pSigma(:, :, k)+1e-5*eye(D));  
%             catch ME
%                 Px(:, k)=1e-8/N;
%                 continue;
%             end;
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);  
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));  
            Px(:, k) = coef * exp(-0.5*tmp);  
        end  
    end  
end  