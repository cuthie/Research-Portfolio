
clc
clear all;

% No of SU
n=2;

% No of Horizon
m=3;

% The sequence of decision to sense
a=reshape((dec2bin(0:2^(n*m)-1)-'0')',n,m,2^(n*m));

% Sequence of Spectrum Sensing result
theta=(dec2bin(0:2^(m)-1)-'0')';

% Time slot interval (100 ms)
T=2;

% Paramters for the distribution of CSI
%  SU-PU Channel
mug=1;

% SU-FC Channel
muh=1;

% Parameters for energy harvesting process (mW)
muH=1;

% Maximum power limit (mW)
P_max=1;

% Spectrum sensing power (mW)
p_s=0.1;

%% Battery state vectors for individual SU

% Maximum Battery power limit (mW)
B_max=0.4:0.2:1.2;

% Number of B_max iteration
b_it=length(B_max);

% Number of Monte carlo iteration
iter=10;

% Number of iteration through random variable H1 and H2
nH=1000;

% Sesning time lower bound (2 ms)
tau_l=0.1;

% Average interference power limit (mW)
Q_avg=1;

% Lagrange Parameter for interfernce
lambda=0.05:0.05:0.2;

% Maximum sum-capacity for non-causal scenario
sum_cap_nc=zeros(iter,b_it);

% Interference for the Non-causal scenario
sum_in_nc=zeros(iter,b_it);

%% Monte carlo loop
for iter_lp=1:iter
    
    % Matrix for SU-PU channel
    g=exprnd(mug,n,m);
    
    % Matrix for SU-FC Channel
    h=exprnd(muh,n,m);
    
    % Matrix for energy harvsting process
    Eng_h=exprnd(muH,n,m);
    
    % Loop for B_max vs throughput plot
    for b_lp=1:b_it
        
        % B_max value for each iteration
        b_m=B_max(b_lp);
        
        % The decision to sense loop
        for l=1:2^(n*m)
            
            % The decision to sense value
            a_it=a(:,:,l);
            
            % The spectrum sensing result loop
            for r=1:2^m
                
                % The sensing result value
                theta_it=theta(:,r);
                
                for l_lp=1:length(lambda)
                    
                    % The value of lambda in the present iteration
                    lamb=lambda(l_lp);
                    
                    tic
                   
                    %% Calling the finite horizon non-causal subroutine
                    [sum_cap_fin_hrz_nc]=fin_hrz_non_cau_dp_mex...
                        (n,m,lamb,a_it,theta_it,tau_l,T,g,h,P_max,b_m,p_s,Eng_h,Q_avg,nH);
                    
                    % Sum capacity for non-causal scenario
                    sum_cap_nc(iter_lp,b_lp)=max(sum_cap_nc(iter_lp,b_lp),sum_cap_fin_hrz_nc);
                    
                    toc
                    
                end
                
                r
                
            end
            
            l
        end
        
        
        b_lp
        
    end
    
    iter_lp
    
end

% Average sum capacity over all iteration
avg_sum_nc=mean(sum_cap_nc,1);


%% Plot
figure
hold on
grid on
plot(B_max,avg_sum_nc,'r*-');
hold off

