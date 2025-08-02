
clc
clear all;

% Horizon Length
m=100000;

% Number of Sensor
n=20;

% Number of Monte Carlo Realizations
iter=35000;

%%  Initialization of Parameters

% Battery State
b_state=zeros(m+1,n);

% Battery State Condition vector
b_state_m=ones(m+1,n);

% CUSUM Test Statistics
s_test=zeros(m+1,n);

% Log-likelihood ratio for H_{0}
llr=zeros(m,n);

% Log-Likelihood ratio for H_{1}
llr_1=zeros(m,n);

% Change point
n_change=zeros(iter,n);

% Normalized Change point
norm_n_change=zeros(iter,n);

% T-min Change Point
n_change_min=zeros(iter,1);

% % T-max Change Point
% n_change_max=zeros(iter,1);

% Gamma parameter
gam=zeros(iter,1);

% Parameter for gamma computation
z_sum=zeros(m+1,1);

% Amount of Energy Harvested
Eng_h=zeros(m,n);

% Energy required for sensing
Es=0.5;

% Energy required for transmission
Eb=1;

% Threshold for the CUSUM test
h=500;

% Initial Battery State
b_state(1,:)=Es+Eb;

% % Initial Battery state condition vector
% b_state_m(1,:)=0;

% Mean of the Observation signal
m_a=0.5;

% Variance of the Observation Signal
var_s=1;

% Mean of LLR
m_llr=m_a^2/(2*var_s);

% Variance of LLR
var_llr=m_a^2/var_s;

% Mean of Individual delay
m_t=h/m_llr;

% Standard Deviation of Individual delay
sig_t=sqrt((h*var_llr)/(m_llr^3));

%% Simulation

% Loop for Monte Carlo Realization
for iter_lp=1:iter
    
    % Sensing Parameter
    mu=zeros(m,n);
    
    % Loop for Time slot
    for k=1:m
        
        % Loop for different sensors
        for i=1:n
            
            % Amount of Energy Harvested
            Eng_h(k,i)=-0.6*log(rand);
            
        end
        
    end
    
    %     % The scenario when E[H] <Es, in that case
    %     % after sufficiently large number of time slots
    %     % b_state(k) > Es
    %     % Transition probabilities are computed via a different program
    %     alphaprob=0.4077;
    %     betaprob=0.6090;
    %
    %     % Loop for time slot
    %     for k=1:m-1
    %
    %         % Loop for different sensors
    %         for i=1:n
    %
    %             % Initialization of random values
    %             u=rand;
    %
    %             if(b_state_m(k,i)==0)
    %
    %                 if(u<=alphaprob)
    %
    %                     b_state_m(k+1,i)=0;
    %
    %                 else
    %
    %                     b_state_m(k+1,i)=1;
    %
    %                 end
    %
    %             else
    %
    %                 if(u<=betaprob)
    %
    %                     b_state_m(k+1,i)=1;
    %
    %                 else
    %
    %                     b_state_m(k+1,i)=0;
    %
    %                 end
    %
    %             end
    %
    %         end
    %
    %     end
    
    
    % The Loop for time slot
    for k=1:m
        
        % The Loop for different sensors
        for i=1:n
            
            % Simulating for null hypothesis
            rv=sqrt(var_s)*randn;
            
            % Log Likelihood ratio for null hypothesis
            llr(k,i)=(rv*m_a/var_s)-(m_a^2/(2*var_s));
            
            % Simulating for non-null hypothesis
            rv_1=m_a+(sqrt(var_s)*randn);
            
            % Log Likelihood ratio for non-null hypothesis
            llr_1(k,i)=(rv_1*m_a/var_s)-(m_a^2/(2*var_s));
            
        end
        
    end
    
    % Initial Change Point
    n_change(iter_lp,:)=0;
    
    % Initial Test Statistics
    s_test(1,:)=0;
    
    % The Loop for different sensors
    for i=1:n
        
        % The Loop for time slot
        for k=1:m
            
            % Checking whether enough energy
            % is available in the battery for sensing
            if(b_state(k,i)>=Es)
                
                % Sensing Parameter modification
                mu(k,i)=1;
                
                % If the CUSUM Test statistics is not
                % above the threshold
                if(s_test(k,i)<=h)
                    
                    % Battery Dynamics
                    b_state(k+1,i)=b_state(k,i)+Eng_h(k,i)-Es;
                    
                    % CUSUM Test statistics modification
                    s_test(k+1,i)=max(0,s_test(k,i)+llr_1(k,i));
                    
                    % Increment Change Point Count
                    n_change(iter_lp,i)=n_change(iter_lp,i)+1;
                    
                else
                    
                    % Battery Dynamics
                    b_state(k+1,i)=b_state(k,i)+Eng_h(k,i)-Es-Eb;
                    
                    % CUSUM Test statistics modification
                    s_test(k+1,i)=max(0,s_test(k,i)+llr_1(k,i));
                    
                    % Change Point Assignment
                    n_change(iter_lp,i)=k;
                    
                    break;
                    
                end
                
            else
                
                % Sensing Parameter modification
                mu(k,i)=0;
                
                % Battery Dynamics
                b_state(k+1,i)=b_state(k,i)+Eng_h(k,i);
                
                % CUSUM Test Statistics modification
                s_test(k+1,i)=max(0,s_test(k,i));
                
                % Increment Change Point Counter
                n_change(iter_lp,i)=n_change(iter_lp,i)+1;
                
            end
            
        end
        
        % Normalized Delay
        norm_n_change(iter_lp,i)=(n_change(iter_lp,i)-m_t)/sig_t;
        
    end
    
    % T-min Change Point
    n_change_min(iter_lp)=min(norm_n_change(iter_lp,:));
    
    %     % T-max Change Point
    %     n_change_max(iter_lp)=max(n_change(iter_lp,:));
    
    %     % Probability factor for T-min
    %     b_prob_min(iter_lp)=(1/n_change_min(iter_lp))*sum(ind([1:n_change_min(iter_lp)]));
    %
    %     % Probability factor for T-max
    %     b_prob_max(iter_lp)=(1/n_change_max(iter_lp))*sum(ind([1:n_change_max(iter_lp)]));
    
    %% Computation of gamma factor via simulaiton
    %
    %     % Initialization of the z_sum paramater
    %     z_sum(1)=0;
    %
    %     % The Loop for time slot
    %     for k=1:m
    %
    %         % Condition on Battery state
    %         if(b_state_m(k,1)==1)
    %
    %             % Random Walk for the z_sum parameter
    %             z_sum(k+1)=z_sum(k)+llr_1(k,1);
    %
    %         else
    %
    %             % No Change Condition
    %             z_sum(k+1)=z_sum(k);
    %
    %         end
    %
    %
    %         % If the parameter exceeds h
    %         if(z_sum(k+1)>h)
    %
    %             % Computation of gamma
    %             gamma(iter_lp)=exp(-(z_sum(k+1)-h));
    %             break;
    %
    %         end
    %
    %     end
    
    %     % The Loop for time slot
    %     for k=1:m
    %
    %         % If the sum exceeds the threshold
    %         if(sum(llr(1:k)) > h)
    %
    %             % Computation of gamma
    %             gamma(iter_lp)=exp(-(sum(llr(1:k))-h));
    %             break;
    %
    %         end
    %
    %     end
    
    iter_lp
    
end

% % Constant gamma in Tartakovsky's Paper
% gam=0.7479;

% Computing gamma via simulation when the change
% has already occured
%gam=(1/iter)*sum(gamma);

%% Computation of the Probability density function of the detection delay
% 
% % espilon term from large deviation results
% epsilon=1;
% 
% % Delay counter
% n_delay=200;
% 
% % Delay threshold paramter
% delay=zeros(n_delay,1);
% 
% % Probability parameter for T-min
% p_delay_min=zeros(n_delay,1);
% 
% % % Probability parameter for T-max
% % p_delay_max=zeros(n_delay,1);
% 
% % Loop for delay counter
% for r=1:n_delay
% 
%     %Computing delay threshold
%     delay(r)=10+80*(r-1)/(n_delay);
%     %delay(r)=10000+50000*(r-1)/(n_delay);
% 
%     % Probability parameter for T-min
%     p_delay_min(r)=0;
% 
% %     % Probability parameter for T-max
% %     p_delay_max(r)=0;
% 
%     % Monte Carlo iteration loop
%     for iter_lp=1:iter
% 
%         % Condition for counting for probability parameter for T-min
%         if(epsilon*n_change_min(iter_lp)>=delay(r))
% 
%             % Incrementing probabilty parameter
%             p_delay_min(r)=p_delay_min(r)+1;
% 
%         end
% 
% %         % Condition for counting for probability parameter for T-min
% %         if(epsilon*n_change_max(iter_lp)>=delay(r))
% %
% %             % Incrementing probabilty parameter
% %             p_delay_max(r)=p_delay_max(r)+1;
% %
% %         end
% 
%     end
% 
% end


% %% Theoretical decay rate of exponential distribution of detection delay
%
% % Exponential term
% lam=((m_a^2)/(2*var_s))*(gam^2/exp(h));
%
% % Distribution for T_min
% ex_prob_min=(n*lam);

% Distribution for T_max

%% Theoretical Mean of the Distribution of the Detection Delay
c=(n*(2^0.5))/(pi^0.5);

% Outer summation variable
s=0;

% Loop for outer summation
for l=0:1:n-1
    
    % Inner coefficient
    e=nchoosek(n-1,l)*((-0.5)^l);
    
    % Inner summation variable
    t=0;
    
    % Loop for inner summation
    for p=1:2:l
        
        t=t+(nchoosek(l,p)*(pi^(-p/2))*(2^p)*gamma((p+2)/2)*Lauricella_A(p));
        
    end
    
    % Calculating outer summation
    s=s+(e*t);
    
end

% The Mean of the theoretical distribution
m_th=c*s;

% The Mean of the simulated distribution
m_ch_min=mean(n_change_min);
        
        


%% Plotting the simulated and theoretical distribution of the detection delay

% figure
% hold on
% grid on
% plot(delay, 1-((1/iter)*p_delay_min),'m*-');
% hold on


% plot(delay, (-epsilon*log((1/iter)*p_delay_max)./delay),'b*-');
% hold on
% yline(ex_prob_min);
%hold off

%Empirical distribution of the runtime to false alarm
[cden,xden]=ksdensity(n_change_min,'Function','pdf');


%% Plotting the distribution

% %Initialization of bins
% xbin=zeros(100,1);
%
% % Pdf Initialization
% cm=zeros(100,1);
% cm=zeros(length(xden),1);

% %Interval Assignment Loop
% for s=1:length(xden)
% 
% %     %Interval Assignment
% %         xbin(s)=(s-1)*(m/1000);
%     %
%     %     % Theoretical pdf
%     %     %fm(s)=(3*lam*exp(-lam*xbin(s)))-(6*lam*exp(-2*lam*xbin(s)))+(3*lam*exp(-3*lam*xbin(s)));
%     %     %fm(s)=exp(-(xbin(s)+exp(-xbin(s))));
%     %     %fm(s)=n*lam*exp(-lam*xbin(s))*((1-(exp(-lam*xbin(s))))^(n-1));
%     %     %fm(s)=n*lam*exp(-lam*xbin(s))*(exp(-n*exp(-lam*xbin(s))));
%     %     fm(s)=lam*exp(-exp((log(n))-(lam*xbin(s))))*exp((log(n))-(lam*xbin(s)));
% 
%     cm(s)=1-((1-(0.5*(1+erf((xden(s)-m_t)/(sqrt(2)*sig_t)))))^n);
% 
% end

% cm=1-((1-normcdf(xden,0,1)).^n);
%cm=1-((1-normcdf(delay,m_t,sig_t)).^n);


% Plotting
figure

hold on
grid on

plot(xden,cden,'m*-');
% hold on
% plot(xden,cm,'k*-');

hold off




