function[mu_opt,r_opt]=opt_thr_obs_pol(m_a,var_s,n,m,mug,muH,thr,b_m,r_max,e_s,g_c,g,Eb_c,Eng_h,Eb,nH)

% The KL information number
% Optimization problem
% Causal CSI and ESI
% Energy Harvesting

%-----------------------------------
% Discrete Variable initialization
%------------------------------------

% Battery level values
b1=0:0.1:b_m;
b2=0:0.1:b_m;

% The number of quantization bits
r1=1:1:r_max;
r2=1:1:r_max;

% The Observation Policy vector
mu=0:1;

% Information vector
I=zeros(m,length(b1),length(b2),length(g_c),length(g_c));

% The Optimal Quantization bits
R1_star=zeros(m,length(b1),length(b2),length(g_c),length(g_c));
R2_star=zeros(m,length(b1),length(b2),length(g_c),length(g_c));

% The Optimal Observation Policy
Mu1_star=zeros(m,length(b1),length(b2),length(g_c),length(g_c));
Mu2_star=zeros(m,length(b1),length(b2),length(g_c),length(g_c));

%% Look Up Table

% Loop for different time slots
for k=m:-1:1
    
    % Loop for Sensor 1 battery state
    for b1_loop=1:length(b1)
        
        % Loop for Sensor 2 battery state
        for b2_loop=1:length(b2)
            
            % Loop for Sensor 1 to FC channel gain
            for g1_loop=1:length(g_c)
                
                % Loop for Sensor 2 to FC channel gain
                for g2_loop=2:length(g_c)
                    
                    % Initialization of Information Number
                    V_max=0;
                    
                    % The Index Initialization
                    max_ind_mu1=1;
                    max_ind_r1=1;
                    max_ind_mu2=1;
                    max_ind_r2=1;
                    
                    % Loop for Sensor 1 observation
                    for ind_mu1=1:length(mu)
                        
                        % Loop for Sensor 2 observation
                        for ind_mu2=1:length(mu)
                            
                            % Loop for Sensor 1 quantization
                            for ind_r1=1:length(r1)
                                
                                % Loop for Sensor 2 quantization
                                for ind_r2=1:length(r2)
                                    
                                    % Number of Quantized State values for Sensor 1
                                    l_it_1=2^(r1(ind_r1));
                                    
                                    % Initialization of distribution
                                    a_1=zeros(1,l_it_1);
                                    b_1=zeros(1,l_it_1);
                                    
                                    % Number of lambda values for Sensor 2
                                    l_it_2=2^r2(ind_r2);
                                    
                                    % Initialization of distribution
                                    a_2=zeros(1,l_it_2);
                                    b_2=zeros(1,l_it_2);
                                    
                                    % Checking if the Energy causality
                                    % Constraint is satisfied for Sensor 1
                                    if(((mu(ind_mu1)*e_s)+(mu(ind_mu1)*r1(ind_r1)*Eb_c(g1_loop)))<=b1(b1_loop))
                                        
                                        % Calling Sub-routine for finding out
                                        % Optimal Threshold for Sensor 1
                                        th_1=thr(r1(ind_r1),1:((2^(r1(ind_r1)))-1));
                                        
                                        % Checking if the Energy causality
                                        % Constraint is satisfied for Sensor 2
                                        if(((mu(ind_mu2)*e_s)+(mu(ind_mu2)*r2(ind_r2)*Eb_c(g2_loop)))<=b2(b2_loop))
                                            
                                            % Calling Sub-routine for finsding out
                                            % Optimal Theshold for Sensor 2
                                            th_2=thr(r2(ind_r2),1:((2^(r2(ind_r2)))-1));
                                            
                                            % Partial Information number
                                            h_1=0;
                                            
                                            % The Loop for Lambda for Sensor 1
                                            for l=1:l_it_1
                                                
                                                % If the first Quantization Level
                                                if(l==1)
                                                    
                                                    a_1(l)=normcdf((th_1(l)-m_a)/sqrt(var_s));
                                                    b_1(l)=normcdf(th_1(l)/sqrt(var_s));
                                                    
                                                    % If the last Quantization Level
                                                elseif(l==l_it_1)
                                                    
                                                    a_1(l)=1-normcdf((th_1(end)-m_a)/sqrt(var_s));
                                                    b_1(l)=1-normcdf((th_1(end))/sqrt(var_s));
                                                    
                                                    % Other Intermediate Quantization
                                                    % Levels
                                                else
                                                    
                                                    a_1(l)=normcdf((th_1(l)-m_a)/sqrt(var_s))-normcdf((th_1(l-1)-m_a)/sqrt(var_s));
                                                    b_1(l)=normcdf(th_1(l)/sqrt(var_s))-normcdf((th_1(l-1))/sqrt(var_s));
                                                    
                                                end
                                                
                                                % Terms for KL divergence
                                                h_1=h_1+((a_1(l)*log(a_1(l)/b_1(l))));
                                                
                                            end
                                            
                                            % Partial Information number
                                            h_2=0;
                                            
                                            % The Loop for Lambda for Sensor 2
                                            for j=1:l_it_2
                                                
                                                % If the first Quantization Level
                                                if(j==1)
                                                    
                                                    a_2(j)=normcdf((th_2(j)-m_a)/sqrt(var_s));
                                                    b_2(j)=normcdf(th_2(j)/sqrt(var_s));
                                                    
                                                    % If the last Quantization Level
                                                elseif(j==l_it_2)
                                                    
                                                    a_2(j)=1-normcdf((th_2(end)-m_a)/sqrt(var_s));
                                                    b_2(j)=1-normcdf((th_2(end))/sqrt(var_s));
                                                    
                                                    % Other Intermediate Quantization
                                                    % Levels
                                                else
                                                    
                                                    a_2(j)=normcdf((th_2(j)-m_a)/sqrt(var_s))-normcdf((th_2(j-1)-m_a)/sqrt(var_s));
                                                    b_2(j)=normcdf(th_2(j)/sqrt(var_s))-normcdf((th_2(j-1))/sqrt(var_s));
                                                    
                                                end
                                                
                                                % Terms for KL divergence
                                                h_2=h_2+((a_2(j)*log(a_2(j)/b_2(j))));
                                                
                                            end
                                            
                                            % Total Information Number
                                            h=(mu(ind_mu1)*h_1)+(mu(ind_mu2)*h_2);
                                            
                                            % Initialization of Value function
                                            tem=0;
                                            
                                            % If not the last slot
                                            if(k<m)
                                                
                                                % Loop for future value
                                                % function simulations
                                                for nxt_ind=1:nH
                                                    
                                                    % Unknown quanties
                                                    H1=exprnd(muH);
                                                    H2=exprnd(muH);
                                                    g1_next=exprnd(mug);
                                                    g2_next=exprnd(mug);
                                                    
                                                    % Battery Level if the control
                                                    % is choosen
                                                    b1_next=min(b_m,b1(b1_loop)+H1-((mu(ind_mu1)*e_s)+...
                                                        (mu(ind_mu1)*r1(ind_r1)*Eb_c(g1_loop))));
                                                    b2_next=min(b_m,b2(b2_loop)+H2-((mu(ind_mu2)*e_s)+...
                                                        (mu(ind_mu2)*r2(ind_r2)*Eb_c(g2_loop))));
                                                    
                                                    % Choosing the appropiate battery index for
                                                    % the next slot
                                                    [~,closest_index_b1]=min(abs(b1_next-b1));
                                                    [~,closest_index_b2]=min(abs(b2_next-b2));
                                                    
                                                    % Choosing the appropiate channel index for the
                                                    % next slot
                                                    [~,closest_index_g1]=min(abs(g1_next-g_c));
                                                    [~,closest_index_g2]=min(abs(g2_next-g_c));
                                                    
                                                    % Calualate the value
                                                    % function if the control
                                                    % is chosen above
                                                    
                                                    % Calculate the value
                                                    % function
                                                    tem=I(k+1,closest_index_b1,closest_index_b2,...
                                                        closest_index_g1,closest_index_g2);
                                                    
                                                end
                                                
                                            end
                                            
                                            % Modified value function
                                            tem=h+(tem/nH);
                                            
                                            % Test if Previous Result is better than the present
                                            % one
                                            if(tem>V_max)
                                                
                                                % In case the test is passed, change the maximum
                                                % index
                                                max_ind_mu1=ind_mu1;
                                                max_ind_r1=ind_r1;
                                                max_ind_mu2=ind_mu2;
                                                max_ind_r2=ind_r2;
                                                
                                                % Reset the V_max
                                                V_max=tem;
                                                
                                            end
                                            
                                        end
                                        
                                    end
                                    
                                end
                                
                            end
                            
                        end
                        
                    end
                    
                    % Save maximal Value of V_max
                    I(k,b1_loop,b2_loop,g1_loop,g2_loop)=V_max;
                    
                    % Save optimal control power input
                    R1_star(k,b1_loop,b2_loop,g1_loop,g2_loop)=r1(max_ind_r1);
                    R2_star(k,b1_loop,b2_loop,g1_loop,g2_loop)=r2(max_ind_r2);
                    
                    % Save optimal control energy transfered
                    Mu1_star(k,b1_loop,b2_loop,g1_loop,g2_loop)=mu(max_ind_mu1);
                    Mu2_star(k,b1_loop,b2_loop,g1_loop,g2_loop)=mu(max_ind_mu2);
                    
                end
                
            end
            
        end
        
    end
    
end

%% Forward Iteration

% Initial battery Level
B1_tem=0.4;
B2_tem=0.4;

% Initialization of the Optimal Observation
mu_opt=zeros(n,m);

% Initialization of the Optimal Quantization
r_opt=zeros(n,m);

% Loop for Simulation
for k=1:1:m
    
    % Finding the closest index correponding to B,g,h for the look up table
    [~,closest_index_B1]=min(abs(B1_tem-b1));
    [~,closest_index_B2]=min(abs(B2_tem-b2));
    
    [~,closest_ind_g1]=min(abs(g(n-1,k)-g_c));
    [~,closest_ind_g2]=min(abs(g(n,k)-g_c));
    
    % Get the optimal values of power from the look up table
    mu1_tem=Mu1_star(k,closest_index_B1,closest_index_B2,closest_ind_g1,closest_ind_g2);
    mu2_tem=Mu2_star(k,closest_index_B1,closest_index_B2,closest_ind_g1,closest_ind_g2);
    
    % Get the optimal values of energy shared from the look up table
    r1_tem=R1_star(k,closest_index_B1,closest_index_B2,closest_ind_g1,closest_ind_g2);
    if(r1_tem==0)
        r1_tem=1;
    end
    
    r2_tem=R2_star(k,closest_index_B1,closest_index_B2,closest_ind_g1,closest_ind_g2);
    
    if(r2_tem==0)
        r2_tem=1;
    end
    
    % Stacking up the Optimal Observation Policy
    mu_opt(:,k)=[mu1_tem;mu2_tem];
    
    % Stacking up the Optimal Quantization Policy
    r_opt(:,k)=[r1_tem;r2_tem];
    
    % Battery state for the Next Slot
    B1_tem=min(b_m,B1_tem-((mu1_tem*e_s)+(mu1_tem*r1_tem*Eb(n-1,k)))+Eng_h(n-1,k));
    B2_tem=min(b_m,B2_tem-((mu2_tem*e_s)+(mu2_tem*r2_tem*Eb(n,k)))+Eng_h(n,k));
    
end

end
