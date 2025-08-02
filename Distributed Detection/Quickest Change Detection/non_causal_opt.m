
% Non-Bayesian
% Decentralized Quickest
% Change Detection
% Framework

clc
clear all;

% Number of Sensors
n=2;

% Number of Time slot
m=8;

% The Unknown Stopping time
tau=0;

% Mean Channel gain
mug=1;

% Mean Harvested Energy
muH=1;

% Number of Energy Harvesting Simulation
nH=10;

% Channel between Sensor and FC
g_c=-mug*log(1-(0.2:0.3:0.8));

% Spectrum Sensing power
e_s=0.1;

% Signal Mean
m_a=-1.5;

% Signal Variance
var_s=1;

% Noise Variance
var_n=0.01;

% Probability of error
Pe=10^-2;

% Energy For Transmitting one Bit
% for different channel gain
Eb_c=(var_n./g_c)*(qfuncinv(2*Pe))^2;

% Pfa limit
gam=4;

% Maximum Battery Limit
B_max=0.4:0.1:1.2;

% Maximum Number of Quantization Points
r_max=5;

% Number of Monte-Carlo Iteration
iter=20;

% Number of B_max iteration
b_it=length(B_max);

% Initialization of Change Point Detection
T=zeros(b_it,iter);

% Initialization of the Observation Signal
x=zeros(n,m);

% Initialization of the CUSUM Test statistics
w=zeros(m,b_it,iter);

% Optimal Obervation
mu_opt=zeros(n,m);
r_opt=zeros(n,m);

% Initialization of the Optimal Theshold
thr=zeros(r_max,(2^r_max)-1);

% Significance Margin
lm=m_a-(3*sqrt(var_s));
um=m_a+(3*sqrt(var_s));

% Calling the Optimal Threshold Subroutine
for l=1:r_max
    
    % Function Handle
    fun=@(th) thr_fun(m_a,var_s,l,th);
    
    % The Number of Threshold
    len_t=2^(l)-1;
    
    % Seperation between Theshold
    delta=(3*sqrt(var_s))/(2^(l-1));
    
    % Initialization of the Threshold
    th_0=lm+delta:delta:um-delta;
    
    % Options for fsolve
    opts=optimoptions('fsolve','Display','off');
    
    % Calling the Fsolve
    th=fsolve(fun,th_0,opts);
    
    % Modifying the Array
    thr(l,1:(2^l)-1)=th;
    
end

%% Monte-Carlo loop
for iter_lp=1:iter
    
    % The channel gain
    g=exprnd(mug,n,m);
    
    % Harvested Energy
    Eng_h=exprnd(muH,n,m);
    
    % Energy For Transmitting one Bit
    Eb=(var_n./g)*(erfcinv(2*Pe))^2;
    
    % Modelling The Observation Signal X_{i,k}
    % Loop for Different time slot
    for k=1:m
        
        % Loop for Different Sensor
        for i=1:n
            
            % Checking the Stopping time
            if(k>tau)
                x(i,k)=normrnd(0,var_n);
            else
                x(i,k)=normrnd(m_a,var_n);
            end
            
        end
        
    end
    
    % The B_max loop
    for b_lp=1:length(B_max)
        
        % The Battery limit
        b_m=B_max(b_lp);
        
        %% The Non-Causal Optimization Subroutine
        [mu_opt,r_opt]=opt_thr_obs_pol_mex(m_a,var_s,n,m,mug,muH,thr,b_m,r_max,e_s,g_c,g,Eb_c,Eng_h,Eb,nH);
        
        %% The CUSUM Procedure Subroutine
        [w(:,b_lp,iter_lp)]=fc_cusum_mex(x,n,m,m_a,var_s,mu_opt,r_opt,thr);
        
        % Stopping time
        % Loop for Time slot
        for k=1:m
            
            if(w(k,b_lp,iter_lp)>=log(gam))
                
                T(b_lp,iter_lp)=k;
                break;
                
            end
            
        end
        
        b_lp
        
    end
    
    iter_lp
    
end

% Average Change Detection Time
T_avg=mean(T,2);

%Plot
figure
hold on
grid on
plot(B_max,T_avg,'b*-');
hold off

