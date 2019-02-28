 function Px = calc_probGaussian(model,X)   
        %Gaussian posterior probability   
        %N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))
        pMiu=model.Miu;
        pSigma=model.Sigma;
        [N,D]=size(X);
        K=size(pMiu,1);
        Px = zeros(N, K);  
        for k = 1:K  
            Xshift = X-repmat(pMiu(k, :), N, 1); %X-pMiu  
            if det(pSigma(:, :, k))<1e-20
                Px(:, k) = 1e-8/N;
                continue;
            end;
            inv_pSigma = inv(pSigma(:, :, k));  
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);  
            coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));  
            Px(:, k) = (1/coef) * exp(-0.5*tmp);  
        end  
end  