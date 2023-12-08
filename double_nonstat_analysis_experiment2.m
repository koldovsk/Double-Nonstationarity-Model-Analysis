Ns = 1000;
d = 10;
K = 5;
L = 5;
T = 1;


N = Ns*L*T;

step = 0.01;
steps = 0.5/step;
trials = 100;

I = eye(K);
CRLB = zeros(steps,steps);
PA = zeros(steps,steps);
EMP = zeros(steps,steps);

sigma2 = ones(1,L);
sigma = sqrt(sigma2);

for idx_true = 1:steps
    for idx_model = 1:steps

        xi_true = 0 + (idx_true-1)*step;
        xi_model = 0 + (idx_model-1)*step;

        Sigma_true = eye(K) + diag(ones(K-1,1)*xi_true,1) + diag(ones(K-1,1)*xi_true,-1);
        Sigma_model = eye(K) + diag(ones(K-1,1)*xi_model,1) + diag(ones(K-1,1)*xi_model,-1);


        %Cz = repmat(eye(d-1), [1 1 K]);
        
        k = 1;
        
        % empirical ISR
        ISR_emp = zeros(trials,1);
        for trial = 1:trials
            s = randn(K,N).*(kron(sigma,ones(K,Ns)));
            s = sqrtm(Sigma_true)*s;
            z = randn(d-1,N,K);
            x = [permute(s,[3 2 1]); -z];
            [w,~,~,NumIt] = fastdiva_test_known_correlation(x,...
                struct('approach','u','ini',repmat(eye(d,1),[1 1 K]),...
                'initype','w','L',L,'nonln','gausstri','known_corr',xi_model));
            isr = ISR_CSV(w,x,[zeros(1,N,K);-z]);
            ISR_emp(trial) = isr(k);
        end
        EMP(idx_true,idx_model) = mean(ISR_emp);
%         psi = Sigma_true\s;
%         phi = Sigma_model\s;
        
        % theoretical ISR
        e = I(:,k);

        kappa = e'*(Sigma_true\e);

        varphi = e'*(Sigma_model\Sigma_true)*(Sigma_model\e);
        if xi_model==xi_true
            nu = 1;
        else
            nu = e'*(Sigma_model\Sigma_true)*e;
        end
        
        rho = e'*(Sigma_model\e);

        CRLB(idx_true,idx_model) = (d-1)/N/mean(sigma2)/...
            (mean(kappa./sigma2)-1/mean(sigma2));

        PA(idx_true,idx_model) = (d-1)/N/mean(sigma2)*...
            (mean(varphi./((nu^2)*sigma2))-1/mean(sigma2))/...
            ((1/mean(sigma2)-mean(rho./(nu*sigma2)))^2);
    
    end
end

figure(1)
subplot(121)
hold off
plot(0:step:0.5-step, real(10*log10(EMP(:,11))),'*')
hold on
plot(0:step:0.5-step, real(10*log10(PA(:,11))));
plot(0:step:0.5-step, real(10*log10(EMP(:,41))),'s')
plot(0:step:0.5-step, real(10*log10(PA(:,41))))
plot(0:step:0.5-step, real(10*log10(CRLB(:,1))))
set(gca,'FontSize',12);
xlabel('\xi','FontSize',16);
ylabel('ISR [dB]','FontSize',16);
legend({'FastDIVA $$\widehat{\xi}$$=0.1', 'FastDIVA $$\widehat{\xi}$$=0.1 teo',...
    'FastDIVA $$\widehat{\xi}$$=0.4', 'FastDIVA $$\widehat{\xi}$$=0.4 teo',...
    'CRiB'},'FontSize',14,'Interpreter','latex');


subplot(122)
hold off
plot(0:step:0.5-step, real(10*log10(EMP(21,:))),'*');
hold on
plot(0:step:0.5-step, real(10*log10(PA(21,:))))
plot(0:step:0.5-step, real(10*log10(CRLB(21,:))))
set(gca,'FontSize',12);
xlabel('$$\widehat{\xi}$$','FontSize',16,'Interpreter','latex');
ylabel('ISR [dB]','FontSize',16);
legend({'FastDIVA', 'FastDIVA teo', 'CRiB'},'FontSize',14,'Interpreter','latex');


