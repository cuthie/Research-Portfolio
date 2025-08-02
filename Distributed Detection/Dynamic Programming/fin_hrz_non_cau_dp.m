function [sum_cap_fin_hrz_nc]=fin_hrz_non_cau_dp(n,m,lambda,a_it,theta_it,tau_l,T,g,h,P_max,b_m,p_s,Eng_h,Q_avg,nH)

% Throughput Maximization problem
% Finite Horizon problem
% Non-Causal Information optimal
% Finite Battery storage
% Energy harevesting and sharing together

%--------------------------------------------------
% Initialization of the discretized variable
%--------------------------------------------------

% Battery state value
b1=0:0.2:b_m;
b2=0:0.2:b_m;

% Discrete power transmission level
step1=P_max/3;
step2=P_max/3;
p1=0:step1:P_max;
p2=0:step2:P_max;

% Discrete sensing time level for SU
t_step=0.95;
t=tau_l:t_step:T;

% Energy sharing level
e12=0:step1:P_max;
e21=0:step2:P_max;

% PU activity probability
mu=0.2;

% Noise variance
n_var=1;

% Interference varaince
inf_var=2;

% Energy transfer efficiency
eta_12=0.8;
eta_21=0.8;

% Value function
V=zeros(m,length(b1),length(b2));

% Optimal value of the transmission power
P1_star=zeros(m,length(b1),length(b2));
P2_star=zeros(m,length(b1),length(b2));

% Optimal Shared energy
e12_star=zeros(m,length(b1),length(b2));
e21_star=zeros(m,length(b1),length(b2));

% Optimal sensing time
t_star=zeros(m,length(b1),length(b2));

% Parameter of the energy harvesting profile
muH1=1;
muH2=1;

% Value function iteration
% Populate the value function for the different horizons
% for SU 1 and 2

% Loop for different horizon
for k=m:-1:1
    
    % Loop for SU1 Battery state
    for b1_loop=1:length(b1)
        
        % Loop for SU2 Battery state
        for b2_loop=1:length(b2)
            
            
            % Initialization of the maximum value function
            V_max=0;
            
            % Initialization of the maximum index
            max_ind_p1=1;
            max_ind_p2=1;
            max_ind_t=1;
            max_ind_e12=1;
            max_ind_e21=1;
            
            % Loop for the SU1 transmit power
            for ind_p1=1:length(p1)
                
                % Loop for the SU2 transmit power
                for ind_p2=1:length(p2)
                    
                    % Loop for the sensing time
                    for ind_t=1:length(t)
                        
                        % Loop for e12 energy sharing
                        for ind_e12=1:length(e12)
                            
                            % Loop for e21 energy sharing
                            for ind_e21=1:length(e21)
                                
                                if(e12(ind_e12)~=0 && e21(ind_e21)~=0)
                                    
                                    continue;
                                    
                                end
                                
                                % Checking energy
                                % causality constraint
                                % for SU 1
                                if(((a_it(n-1,k)*(p_s*t(ind_t)+(p1(ind_p1)*(T-t(ind_t))*(1-theta_it(k)))))...
                                        -e12(ind_e12)+(eta_21*e21(ind_e21))) <=b1(b1_loop))
                                    
                                    
                                    % Checking energy
                                    % causality constraint
                                    % for SU 2
                                    if(((a_it(n,k)*(p_s*t(ind_t)+(p2(ind_p2)*(T-t(ind_t))*(1-theta_it(k)))))...
                                            -e21(ind_e21)+(eta_12*e12(ind_e12))) <=b2(b2_loop))
                                        
                                        
                                        %  The Capacity
                                        %  expression for
                                        %  the present slot
                                        
                                        % SU1
                                        cp1_term=p1(ind_p1)*h(n-1,k)*a_it(n-1,k)*(1-theta_it(k));
                                        
                                        % SU2
                                        cp2_term=p2(ind_p2)*h(n,k)*a_it(n,k)*(1-theta_it(k));
                                        
                                        % Present slot sum
                                        % capacity
                                        % expression when
                                        % PU is present
                                        cp_term_p=(mu/m)*((T-t(ind_t))/T)*(1/log(2))*log(1+((cp1_term+cp2_term)...
                                            /(n_var+inf_var)));
                                        
                                        % Present slot sum
                                        % capacity
                                        % expression when
                                        % PU is absent
                                        cp_term_ab=((1-mu)/m)*((T-t(ind_t))/T)*(1/log(2))*log(1+((cp1_term+cp2_term)...
                                            /n_var));
                                        
                                        % Total term
                                        cp_term=cp_term_p+cp_term_ab;
                                        
                                        % Interference
                                        % component
                                        
                                        %SU1
                                        in1_term=p1(ind_p1)*g(n-1,k)*a_it(n-1,k)*(1-theta_it(k));
                                        
                                        % SU2
                                        in2_term=p2(ind_p2)*g(n,k)*a_it(n,k)*(1-theta_it(k));
                                        
                                        % Present slot
                                        % interfernce
                                        in_term=lambda*(((mu)/m)*(((T-t(ind_t))/T)*(in1_term+in2_term))-Q_avg);
                                        
                                        % Modified
                                        % Objective
                                        % function
                                        obj_term=cp_term-in_term;
                                        
                                        % Intializing the
                                        % value function
                                        tem=0;
                                        
                                        % If not the last slot
                                        if(k<m)
                                            
                                            % Loop for the
                                            % Energy harvesting
                                            for H_ind=1:nH
                                                
                                                % The random
                                                % parameters
                                                H1=exprnd(muH1);
                                                H2=exprnd(muH2);
                                                
                                                % Battery
                                                % levels if
                                                % the controls
                                                % are choosen
                                                b1_next=min(b_m,b1(b1_loop)+H1-a_it(n-1,k)*(p_s*t(ind_t)+(p1(ind_p1)*...
                                                    (T-t(ind_t))*(1-theta_it(k))))-e12(ind_e12)+(eta_21*e21(ind_e21)));
                                                b2_next=min(b_m,b2(b2_loop)+H2-a_it(n,k)*(p_s*t(ind_t)+(p2(ind_p2)*...
                                                    (T-t(ind_t))*(1-theta_it(k))))-e21(ind_e21)+(eta_12*e12(ind_e12)));
                                                
                                                % Choose the
                                                % appropiate
                                                % battery bins
                                                % for the next
                                                % slot
                                                [~,closest_index_b1]=min(abs(b1_next-b1));
                                                [~,closest_index_b2]=min(abs(b2_next-b2));
                                                
                                                % Calculate the
                                                % value
                                                % function if
                                                % the control
                                                % is choosen
                                                % above
                                                
                                                % Calculate the
                                                % value
                                                % function
                                                tem=tem+V(k+1,closest_index_b1,closest_index_b2);
                                                
                                            end
                                            
                                        end
                                        
                                        % Modified value
                                        % function
                                        tem=obj_term+(tem/nH);
                                        
                                        % Test if the
                                        % result is better
                                        % than the previous
                                        % case
                                        if(tem>V_max)
                                            
                                            % Test
                                            % passed:reset
                                            % index
                                            max_ind_p1=ind_p1;
                                            max_ind_p2=ind_p2;
                                            max_ind_t=ind_t;
                                            max_ind_e12=ind_e12;
                                            max_ind_e21=ind_e21;
                                            
                                            % Reset the
                                            % maximum value
                                            % function
                                            V_max=tem;
                                            
                                        end
                                        
                                    end
                                    
                                end
                                
                            end
                            
                        end
                        
                    end
                    
                end
                
            end
            
            % Save maximum value function
            V(k,b1_loop,b2_loop)=V_max;
            
            % Save optimal control power input
            P1_star(k,b1_loop,b2_loop)=p1(max_ind_p1);
            P2_star(k,b1_loop,b2_loop)=p2(max_ind_p2);
            
            % Save optimal control sensing time input
            t_star(k,b1_loop,b2_loop)=t(max_ind_t);
            
            % Save optimal energy transfer
            e12_star(k,b1_loop,b2_loop)=e12(max_ind_e12);
            e21_star(k,b1_loop,b2_loop)=e21(max_ind_e21);
            
        end
        
    end
    
end


% Intialize the vectors for the simulation
% Batthery level vectors
b1_sim=zeros(1,m+1);
b2_sim=zeros(1,m+1);

% Energy usage vectors
p1_sim=zeros(1,m);
p2_sim=zeros(1,m);

% Energy sharing vectors
e12_sim=zeros(1,m);
e21_sim=zeros(1,m);

% Throughput vector
sum_cap_fin_hrz_nc=0;

% Sensing time vector
t_sim=zeros(1,m);

% set initial values for time simulation
% initial battery levels
b1_sim(1)=0.4;
b2_sim(1)=0.4;

B1_tem=b1_sim(1);
B2_tem=b2_sim(1);

% Loop for the Simulation
for k=1:1:m
    
    % Finding the closest index correponding to B,g,h for the look up table
    [~,closest_index_B1] = min(abs(B1_tem-b1));
    [~,closest_index_B2] = min(abs(B2_tem-b2));
    
    % Get the optiomal values of sensing time from look up table
    t_tem=t_star(k,closest_index_B1,closest_index_B2);
    
    % Get the optimal value of the energy transfered form the look up table
    e12_tem=min(e12_star(k,closest_index_B1,closest_index_B2),B1_tem);
    e21_tem=min(e21_star(k,closest_index_B1,closest_index_B2),B2_tem);
    
    % Modified Power limit
    tp1_lim=((B1_tem-(p_s*t_tem)+e12_tem-(eta_21*e21_tem))/(T-t_tem));
    tp2_lim=((B2_tem-(p_s*t_tem)+e21_tem-(eta_12*e12_tem))/(T-t_tem));
    
    % Get the optimal values of power from the look up table
    p1_tem=min(P1_star(k,closest_index_B1,closest_index_B2),tp1_lim);
    p2_tem=min(P2_star(k,closest_index_B1,closest_index_B2),tp2_lim);
    
    % Calculate the Throughput
    cp1_tem=p1_tem*h(n-1,k)*a_it(n-1,k)*(1-theta_it(k));
    cp2_tem=p2_tem*h(n,k)*a_it(n,k)*(1-theta_it(k));
    
    % Total Throughput
    cp_tem_p=(mu/m)*((T-t_tem)/T)*(1/log(2))*log(1+((cp1_tem+cp2_tem)/(n_var+inf_var)));
    cp_tem_ab=((1-mu)/m)*((T-t_tem)/T)*(1/log(2))*log(1+((cp1_tem+cp2_tem)/n_var));
    cp_tem=cp_tem_p+cp_tem_ab;
    
    % Battery state for the next slot
    B1_tem=min(b_m,B1_tem+Eng_h(n-1,k)-(a_it(n-1,k)*(p_s*t_tem+(p1_tem*(T-t_tem)*(1-theta_it(k)))))...
        -e12_tem+(eta_21*e21_tem));
    B2_tem=min(b_m,B2_tem+Eng_h(n,k)-(a_it(n,k)*(p_s*t_tem+(p1_tem*(T-t_tem)*(1-theta_it(k)))))...
        -e21_tem+(eta_12*e12_tem));
    
    
    % Optimal value storage
    b1_sim(k+1)=B1_tem;
    b2_sim(k+1)=B2_tem;
    p1_sim(k)=p1_tem;
    p2_sim(k)=p2_tem;
    t_sim(k)=t_tem;
    e12_sim(k)=e12_tem;
    e21_sim(k)=e21_tem;
    sum_cap_fin_hrz_nc=sum_cap_fin_hrz_nc+cp_tem;
    
end

end
