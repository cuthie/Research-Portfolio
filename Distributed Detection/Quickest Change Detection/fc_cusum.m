function[w]=fc_cusum(x,n,m,m_a,var_s,mu_opt,r_opt,thr)

% The CUSUM test in the
% Fusion Center

% Test Statistics Initialization
z=zeros(1,m);
w=zeros(1,m);

% Initialization of LLR
q=zeros(n,m);

%% The Time slot Loop
for k=1:m
    
    % The Log-Likelihood ratio for Sensor 1
    q(n-1,k)=(m_a*x(n-1,k)/var_s)-((m_a^2)/(2*var_s));
    
    % Initialization
    z_1=0;
    z_2=0;
    
    % Calling Sub-routine for finding out
    % Optimal Threshold for Sensor 1
    th_1=thr(r_opt(n-1,k),1:((2^(r_opt(n-1,k)))-1));
    
    % Observation Condition
    if(mu_opt(n-1,k)==1)
        
        % The lowest Quantization point
        if(q(n-1,k)<=th_1(1))
            
            % The Distribution Terms
            a_1=normcdf((th_1(1)-m_a)/sqrt(var_s));
            b_1=normcdf(th_1(1)/sqrt(var_s));
            
            % Partial Statistics
            z_1=log(a_1/b_1);
            
            % The Highest Quantized Point
        elseif(q(n-1,k)>th_1(end))
            
            % The Distribution Terms
            a_1=1-normcdf((th_1(end)-m_a)/sqrt(var_s));
            b_1=1-normcdf(th_1(end)/sqrt(var_s));
            
            % Partial Statistics
            z_1=log(a_1/b_1);
            
            % All other Quantization Points
        else
            
            % Sensor 1 quantized value
            for th=1:(2^(r_opt(n-1,k)))-2
                
                % Condition for the U
                if(q(n-1,k)>th_1(th) && q(n-1,k)<=th_1(th+1))
                    
                    % The Distribution Terms
                    a_1=normcdf((th_1(th+1)-m_a)/sqrt(var_s))-normcdf((th_1(th)-m_a)/sqrt(var_s));
                    b_1=normcdf(th_1(th+1)/sqrt(var_s))-normcdf((th_1(th))/sqrt(var_s));
                    
                    % Partial Statistics
                    z_1=log(a_1/b_1);
                    break;
                    
                end
                
            end
            
        end
        
    end
    
    % Calling Sub-routine for finding out
    % Optimal Threshold for Sensor 1
    th_2=thr(r_opt(n,k),1:((2^(r_opt(n,k)))-1));
 
    % The Log-Likelihood ratio for Sensor 2
    q(n,k)=(m_a*x(n,k)/var_s)-((m_a^2)/(2*var_s));
    
    % Observation Condition
    if(mu_opt(n,k)==1)
        
        % The lowest Quantization point
        if(q(n,k)<=th_2(1))
            
            % The Distribution Terms
            a_2=normcdf((th_2(1)-m_a)/sqrt(var_s));
            b_2=normcdf(th_2(1)/sqrt(var_s));
            
            % Partial Statistics
            z_2=log(a_2/b_2);
            
            % The Highest Quantized Points
        elseif(q(n,k)>th_2(end))
            
            % The Distribution Terms
            a_2=1-normcdf((th_2(end)-m_a)/sqrt(var_s));
            b_2=1-normcdf(th_2(end)/sqrt(var_s));
            
            % Partial Statistics
            z_2=log(a_2/b_2);
            
            % All other quantization Points
        else
            
            % Sensor 1 quantized value
            for th=1:(2^(r_opt(n,k)))-2
                
                % Condition for the U
                if(q(n,k)>th_2(th) && q(n,k)<=th_2(th+1))
                    
                    % The Distribution Terms
                    a_2=normcdf((th_2(th+1)-m_a)/sqrt(var_s))-normcdf((th_2(th)-m_a)/sqrt(var_s));
                    b_2=normcdf(th_2(th+1)/sqrt(var_s))-normcdf((th_2(th))/sqrt(var_s));
                    % Partial Statistics
                    z_2=log(a_2/b_2);
                    break;
                    
                end
                
            end
            
        end
        
    end
    
    % Sum CUSUM statistics
z(k)=z_1+z_2;
    
    % The First Time Slot
   if(k==1)
        
        % Recursive definition
        w(k)=max(0,z(k));
        
    else
        
        % Recursive definition
        w(k)=max(0,w(k-1)+z(k));
        
    end
    
end

w

end
