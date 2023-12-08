clear *
load hlasy

d = 6;
N = 20000;
L = 20;
trials = 1000;
params = [500 800 1000 2500 5000 10000 20000];
variants = length(params);


ISRef = zeros(trials,variants);
ISRef_teo = zeros(trials,variants);
ISRfdiva = zeros(trials,variants);
ISRfdiva_teo = zeros(trials,variants);  
ISRfdivarati = zeros(trials,variants);
ISRfdivarati_teo = zeros(trials,variants);
ISRfdivaratistat = zeros(trials,variants);
ISRfdivaratistat_teo = zeros(trials,variants);


for variant = 1:variants
    
N = params(variant);

for trial = 1:trials

    ind = ceil(rand*(length(S)-N));
    s = S(randperm(15,1),ind:ind+N-1);
    z = randn(d-1,N);
    A = randn(d);
    noise = A(:,2:end)*z;
    W = inv(A);
    x = A*[s;z];
    
    % EFICA 
    
    [Wef, isr_teo] = efica(x, W);
    G = abs(Wef*A).^2;
    [~, ind] = max(G(:,1)./sum(G(:,2:end),2));
    ISRef(trial,variant) = ISR_CSV(Wef(ind,:)', x, noise);
    ISRef_teo(trial,variant) = sum(isr_teo(ind,:));
    
    % FastDIVA gauss
    
    [w, a] = fastdiva(x,struct('initype', 'w', 'ini', W(1,:)', 'nonln', 'gauss', 'L', L));
    ISRfdiva(trial,variant) = ISR_CSV(w, x, noise);
    
    Ns = N/L;
    soi = w'*x;
    sigma2 = zeros(1,1,L);
    kappa = zeros(1,1,L);
    Cz = zeros(d-1,d-1,L);
    for ell = 1:L
        subblock = (ell-1)*Ns+1:ell*Ns;
        sigma2(1,1,ell) = mean(abs(soi(subblock)).^2);
        B = [a(2:end) -a(1)*eye(d-1)];
        Cz(:,:,ell) = B*(x(:,subblock)*x(:,subblock)')*B'/Ns;
        kappa(1,1,ell) = 1/sigma2(1,1,ell);
    end
    
    aux = mean(Cz.*sigma2,3)/(mean(sigma2,3)^2) + ...
           mean(Cz.*kappa,3) - ...
            2*mean(Cz,3)/mean(sigma2,3);
    aux2 = mean(Cz,3)/mean(sigma2,3) - mean(Cz.*kappa,3);
    aux2 = inv(aux2);
    PerAn = aux2*aux*aux2'/N;
    ISRfdiva_teo(trial,variant) = trace((mean(Cz,3)*PerAn)/mean(sigma2,3));

    % FastDIVA rati L=20
    
    [w, a] = fastdiva(x,struct('initype', 'w', 'ini', W(1,:)', 'nonln', 'rati', 'L', L));
    ISRfdivarati(trial,variant) = ISR_CSV(w, x, noise);
    
    Ns = N/L;
    soi = w'*x;
    sigma2 = zeros(1,1,L);
    phi = zeros(1,1,L);
    nu = zeros(1,1,L);
    rho = zeros(1,1,L);
    Cz = zeros(d-1,d-1,L);
    for ell = 1:L
        subblock = (ell-1)*Ns+1:ell*Ns;
        sigma2(1,1,ell) = mean(abs(soi(subblock)).^2);
        nsoi = soi(subblock)/sqrt(sigma2(1,1,ell));
        B = [a(2:end) -a(1)*eye(d-1)];
        Cz(:,:,ell) = B*(x(:,subblock)*x(:,subblock)')*B'/Ns;
        sp2 = nsoi.*conj(nsoi);
        aux = 1./(1+sum(sp2,3));
        psi = conj(nsoi).*aux;
        psipsi = aux - 2*psi.*conj(psi); 
        phi(1,1,ell) = mean(abs(psi).^2);
        nu(1,1,ell) = mean(psi.*nsoi);
        rho(1,1,ell) = mean(psipsi);
    end

    aux = mean(Cz.*sigma2,3)/(mean(sigma2,3)^2) + ...
       mean(Cz.*phi./sigma2./(abs(nu).^2),3) - 2*mean(Cz,3)/mean(sigma2,3);
    aux2 = mean(Cz,3)/mean(sigma2,3) - mean(Cz.*rho./sigma2./nu,3);
    aux2 = inv(aux2);
    PerAn = aux2*aux*aux2'/N;
    ISRfdivarati_teo(trial,variant) = trace((mean(Cz,3)*PerAn)/mean(sigma2,3));

    % FastDIVA rati L=1
    
    [w, a] = fastdiva(x,struct('initype', 'w', 'ini', W(1,:)', 'nonln', 'rati', 'L', 1));
    ISRfdivaratistat(trial,variant) = ISR_CSV(w, x, noise);
    
    soi = w'*x;
    sigma2stat = mean(abs(soi).^2);
    nsoi = soi/sqrt(sigma2stat);
    sp2 = nsoi.*conj(nsoi);
    aux = 1./(1+sum(sp2,3));
    psi = conj(nsoi).*aux;
    psipsi = aux - 2*psi.*conj(psi); 
    phistat = mean(abs(psi).^2);
    nustat = mean(psi.*nsoi);
    rhostat = mean(psipsi);

    ISRfdivaratistat_teo(trial,variant) = (phistat-nustat^2)/((rhostat-nustat)^2)/N;


end

end

figure
plot(params,10*log10(trimmean(ISRef,5)),'linewidth',2,'Marker','o','LineStyle','none');
hold on
plot(params,10*log10(trimmean(ISRef_teo,5)),'linewidth',2,'LineStyle',':')
plot(params,10*log10(trimmean(ISRfdivaratistat,5)),'linewidth',2,'Marker','s','LineStyle','none')
plot(params,10*log10(trimmean(ISRfdivaratistat_teo,5)),'linewidth',2,'LineStyle','-.')
plot(params,10*log10(trimmean(ISRfdivarati,5)),'linewidth',2,'Marker','^','LineStyle','none')
plot(params,10*log10(trimmean(ISRfdivarati_teo,5)),'linewidth',2,'LineStyle','--')
plot(params,10*log10(trimmean(ISRfdiva,5)),'linewidth',2,'Marker','p','LineStyle','none')
plot(params,10*log10(trimmean(ISRfdiva_teo,5)),'linewidth',2)
set(gca,'FontSize',12);
legend({'EFICA', 'EFICA teo', 'FastDIVA rati L=1', 'FastDIVA rati L=1 teo',...
    'FastDIVA rati L=20', 'FastDIVA rati L=20 teo',...
    'FastDIVA gauss L=20', 'FastDIVA gauss L=20 teo'},'FontSize',14);
xlabel('N (# samples)','FontSize',16);
ylabel('ISR [dB]','FontSize',16);


