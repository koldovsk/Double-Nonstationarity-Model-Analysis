function double_nonstat_analysis_experiment1(domain,ndist)


if nargin<1
    domain = 'complex';
    ndist = 'normal';
end
%%%%%%%%%% Data

T = 3; % # blocks
L = 5; % # subblocks
d = 6; % # sources
r = d; % # channels
K = 1; % # modes (the number of models, ICA... = 1, IVA... > 1 )
minvar = 0.01;
maxvar = 1;
SIRinlow = 0;
SIRinhigh = 0;
circularitycoef = 0.5;
%alpha = 1;
sigma = 0.01;
alpha = 2;


%%%%%%%%%% Methods
%maxIter = 100;
numMethods = 13;


%%%%%%%%%% Simulation
ntrials = 1000;
% profile clear
% profile on 
parameters = [10 25 50 100 200 500 1000 10000]*15; % 2:2:10];

%%%%%%%%%%% Outputs
itertime = zeros(ntrials,length(parameters),K,numMethods);
iterations = zeros(ntrials,length(parameters),K,numMethods);
oISR = zeros(ntrials,length(parameters),K,numMethods);
iISR = zeros(ntrials,length(parameters),K);



for ind_param = 1:length(parameters)

N = parameters(ind_param); % # samples in a block
Nb = N/T;
Ns = Nb/L;


ini_variances = zeros(d,N,K);
subvariances = zeros(d,Nb,K);
for k = 1:K    
    ini_variances(1,:,k) = kron(sin((1:T)/(T+1)*pi).^(alpha/2),ones(1,Nb));
    ini_variances(2:end,:,k) = kron(sqrt(minvar + (maxvar-minvar)*rand(d-1,T)),ones(1,Nb));
    
    subvariances(1,:,k) = kron(abs(sin((1:L)/(L+1)*pi)).^(alpha/2),ones(1,Ns));
    subvariances(2:end,:,k) = ones(d-1,Nb);
end

ini_variances = ini_variances.*kron(subvariances,ones(1,T));

if strcmp(domain,'real')
    U = orth(randn(K));
elseif strcmp(domain,'complex')
    U = orth(crandn(K));
end
    
for trial = 1:ntrials %parfor
    
trialresults = zeros(4,K,numMethods);
%ISRtrial = zeros(K,maxIter,numMethods);
%timetrial = zeros(maxIter,numMethods);

disp(['parameter index:' num2str(ind_param) ' trial:' num2str(trial)]);

%% Sources
s = zeros(d,N,K);

if strcmp(domain,'real')
    aux = laplace(K,N);
    s(1,:,:) = permute(U*(permute(ini_variances(1,:,:),[3 2 1]).*aux),[3 2 1]);
    if strcmp(ndist,'normal')
        s(2:d,:,:) = ini_variances(2:d,:,:).*randn(d-1,N,K); 
    else
        s(2:d,:,:) = ini_variances(2:d,:,:).*laplace(d-1,N,K); 
    end
elseif strcmp(domain,'complex')
    aux = zeros(K,N);
    for i = 1:K
       aux(i,:) = cggd_rand(1,1,N,circularitycoef); 
    end
    s(1,:,:) = permute(U*(permute(ini_variances(1,:,:),[3 2 1]).*aux),[3 2 1]);
    if strcmp(ndist,'normal')
        s(2:d,:,:) = ini_variances(2:d,:,:).*crandn(d-1,N,K); 
    else
        s(2:d,:,:) = ini_variances(2:d,:,:).*claplace(d-1,N,K); 
    end
end

powers = squeeze(sum(s.*conj(s),2));
powers = [powers(1,:); mean(powers(2:end,:),1)];
inputISR = powers(2,:)./powers(1,:);
setSR = SIRinlow + (SIRinhigh-SIRinlow)*rand(1,K); %double(rand(1,M)>0.5);
gain = sqrt(inputISR.*10.^(setSR/10)); 
s(1,:,:) = permute(gain,[1 3 2]).*s(1,:,:);

variances = zeros(d,N,K);
variances(1,:,:) = (ini_variances(1,:,:)*gain).^2;
variances(2:d,:,:) = 2*ini_variances(2:d,:,:).^2;

%% Mixing matrices
if strcmp(domain,'real')
    Wtrue = 1+rand(r,d,K,T);
elseif strcmp(domain,'complex')
    Wtrue = 1+rand(r,d,K,T)+1i*rand(r,d,K,T);
end
Wtrue(1:1,:,:,:) = repmat(Wtrue(1:1,:,:,1),1,1,1,T);
Atrue = zeros(size(Wtrue));
for k = 1:K
    for t = 1:T
        Atrue(:,:,k,t) = inv(Wtrue(:,:,k,t));
    end
end
x = zeros(r,N,K);
noise = zeros(r,N,K);

IVEwini = zeros(d,K);

trial_iISR = zeros(K,1);



for k = 1:K
% Observations
A = squeeze(Atrue(:,:,k,:)); 
sk = s(:,:,k);
nsoi = sk(1,:)./sqrt(variances(1,:));
xk = zeros(r,N);
Cx_true = zeros(d,d,T,L);
Cz_true = zeros(d-1,d-1,T,L);
kappa = zeros(1,1,T,L);
phi = zeros(1,1,T,L);
nu = zeros(1,1,T,L);
rho = zeros(1,1,T,L);
svar = zeros(1,1,T,L);
phiblock = zeros(1,1,T);
nublock = zeros(1,1,T);
rhoblock = zeros(1,1,T);
for t = 1:T
    block = (t-1)*Nb+1:t*Nb;
    xk(:,(t-1)*Nb+1:t*Nb) = A(:,:,t)*sk(:,(t-1)*Nb+1:t*Nb);
    noise(:,(t-1)*Nb+1:t*Nb,k) = A(:,2:end,t)*sk(2:end,(t-1)*Nb+1:t*Nb);
    for ell = 1:L
        subblock = block((ell-1)*Ns+1:ell*Ns);
        svar(1,1,t,ell) = variances(1,subblock(1));
        Cx_true(:,:,t,ell) = A(:,:,t)*diag(variances(:,subblock(1)))*A(:,:,t)';
        B = [A(2:end,1,t) -A(1,1,t)*eye(d-1)];
        Cz_true(:,:,t,ell) = B*Cx_true(:,:,t,ell)*B';
        kappa(1,1,t,ell) = 1/(svar(1,1,t,ell)*(1-abs(circularitycoef)^2));
        sp2 = nsoi(subblock).*conj(nsoi(subblock));
        aux = 1./(1+sum(sp2,3));
        psi = conj(nsoi(subblock)).*aux;
        psipsi = aux - psi.*conj(psi); 
        phi(1,1,t,ell) = mean(abs(psi).^2);
        nu(1,1,t,ell) = mean(psi.*nsoi(subblock));
        rho(1,1,t,ell) = mean(psipsi);
    end
    nsoiblock = sk(1,block)/sqrt(mean(variances(1,block)));
    sp2 = nsoiblock.*conj(nsoiblock);
    aux = 1./(1+sum(sp2,3));
    psi = conj(nsoiblock).*aux;
    psipsi = aux - psi.*conj(psi); 
    phiblock(1,1,t) = mean(abs(psi).^2);
    nublock(1,1,t) = mean(psi.*nsoiblock);
    rhoblock(1,1,t) = mean(psipsi);
end

x(:,:,k) = xk;

powers = real(mean(sk.*conj(sk),2));
trial_iISR(k) = mean(powers(2:end))./powers(1);

%% Testing

% initialization
if strcmp(domain,'real')
    difference = randn(r,1);
else
    difference = crandn(r,1);
end
wtrue = Wtrue(1,:,k,1)';
difference = difference-(wtrue'*difference)/(wtrue'*wtrue)*wtrue;
difference = difference/norm(difference)*sqrt(sigma);
wini = wtrue + difference;
Cx = xk*xk'/N;
aux = Cx*wini;
aini = aux/(wini'*aux);
Wini = [wini'; [aini(2:end) -aini(1)*eye(d-1)]];

for method = 1:9 % ICE/ICA
    tic
    w = wini;
    NumIt = 0;
    switch method
        case 1 % Perf. Analysis nongauss 
            aux = zeros(d-1,d-1);
            aux2 = zeros(d-1,d-1);
            for t = 1:T
                aux = aux + ...
                   mean(Cz_true(:,:,t,:),4).*phiblock(1,1,t)./mean(svar(1,1,t,:),4)./(abs(nublock(1,1,t)).^2) - ...
                    mean(Cz_true(:,:,t,:),4)/mean(svar(1,1,t,:),4);
                aux2 = aux2 + mean(Cz_true(:,:,t,:),4)/mean(svar(1,1,t,:),4) - ...
                    mean(Cz_true(:,:,t,:),4).*rhoblock(1,1,t)./mean(svar(1,1,t,:),4)./nublock(1,1,t);
            end
            aux = aux/T;
            aux2 = inv(aux2/T);
            PerAn = aux2*aux*aux2'/N;
            PerAn = real(trace((mean(Cz_true,[3 4])*PerAn)/mean(svar,[3 4])));
        case 2 % Perf. Analysis nongauss nonstat
            aux = zeros(d-1,d-1);
            aux2 = zeros(d-1,d-1);
            for t = 1:T
                aux = aux + mean(Cz_true(:,:,t,:).*svar(1,1,t,:),4)/...
                   (mean(svar(1,1,t,:),4)^2) + ...
                   mean(Cz_true(:,:,t,:).*phi(1,1,t,:)./svar(1,1,t,:)./(abs(nu(1,1,t,:)).^2),4) - ...
                    2*mean(Cz_true(:,:,t,:),4)/mean(svar(1,1,t,:),4);
                aux2 = aux2 + mean(Cz_true(:,:,t,:),4)/mean(svar(1,1,t,:),4) - ...
                    mean(Cz_true(:,:,t,:).*rho(1,1,t,:)./svar(1,1,t,:)./nu(1,1,t,:),4);
            end
            aux = aux/T;
            aux2 = inv(aux2/T);
            PerAn = aux2*aux*aux2'/N;
            PerAn = real(trace((mean(Cz_true,[3 4])*PerAn)/mean(svar,[3 4])));
        case 3 % CRLB
            aux = zeros(d-1,d-1);
            for t = 1:T
                aux = aux + mean(Cz_true(:,:,t,:).*kappa(1,1,t,:),4)-...
                    inv(mean(svar(1,1,t,:).*pageinv(Cz_true(:,:,t,:)),4));
            end
            aux = inv(aux/T);
            CRiB = real(trace((mean(Cz_true,[3 4])*aux)/mean(svar,[3 4]))/N);
            
        case 4 % FastDIVA nongauss
            [w, ~, ~, NumIt] = fastdiva(xk, struct('ini', wini, 'initype', 'w', 'T', T, 'L', 1, 'nonln', 'rati'));
            w = w(:,1,1,1);
        case 5 % FastDIVA nongauss nonstat
            [w, ~, ~, NumIt] = fastdiva(xk, struct('ini', wini, 'initype', 'w', 'T', T, 'L', L, 'nonln', 'rati'));
            w = w(:,1,1,1);
        case 6 % FastDIVA gauss nonstat
            [w, ~, ~, NumIt] = fastdiva(xk, struct('ini', wini, 'initype', 'w', 'T', T, 'L', L, 'nonln', 'gauss'));
            w = w(:,1,1,1);
        case 7 % Perf. Analysis gauss nonstat
            aux = zeros(d-1,d-1);
            aux2 = zeros(d-1,d-1);
            for t = 1:T
                aux = aux + mean(Cz_true(:,:,t,:).*svar(1,1,t,:),4)/...
                   (mean(svar(1,1,t,:),4)^2) + ...
                   mean(Cz_true(:,:,t,:).*kappa(1,1,t,:),4) - ...
                    2*mean(Cz_true(:,:,t,:),4)/mean(svar(1,1,t,:),4);
                aux2 = aux2 + mean(Cz_true(:,:,t,:),4)/mean(svar(1,1,t,:),4) - ...
                    mean(Cz_true(:,:,t,:).*kappa(1,1,t,:),4);
            end
            aux = aux/T;
            aux2 = inv(aux2/T);
            PerAn = aux2*aux*aux2'/N;
            PerAn = real(trace((mean(Cz_true,[3 4])*PerAn)/mean(svar,[3 4])));
    end
    resultingISR = ISR_CSV(w,xk,noise(:,:,k));
    if method == 1, resultingISR = PerAn; end
    if method == 2, resultingISR = PerAn; end
    if method == 3, resultingISR = CRiB; end
    if method == 7, resultingISR = PerAn; end
    trialresults(1,k,method) = toc;
    trialresults(2,k,method) = NumIt;
    trialresults(3,k,method) = resultingISR;
end

IVEwini(:,k) = wini;
end






itertime(trial,ind_param,:,:) = permute(trialresults(1,:,:),[1 4 2 3]);
iterations(trial,ind_param,:,:) = permute(trialresults(2,:,:),[1 4 2 3]);
oISR(trial,ind_param,:,:) = permute(trialresults(3,:,:),[4 1 2 3]);
iISR(trial,ind_param,:) = trial_iISR;
end

end

%save TSPexperimentCSV_1_analysis

figure
h = axes; %subplot(121);
aux = 10*log10(squeeze(trimmean(oISR(:,:,:,[4 1 5 2 6 3]),5)));
semilogx(parameters,aux, 'linewidth', 2);
set(h,'FontSize',12);
legend({'FastDIVA rati L=1', 'FastDIVA rati L=1 teo', 'FastDIVA rati L=5', 'FastDIVA rati L=5 teo',...
    'FastDIVA gauss L=5', 'CRiB'},'FontSize',14); %, 'FITJBD', 'SIMJBD')
xlabel('N (# samples)','FontSize',16);
ylabel('ISR [dB]','FontSize',16);

