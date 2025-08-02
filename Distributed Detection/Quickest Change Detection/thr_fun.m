function[F]=thr_fun(m_a,var_s,r,th)

% The Number of Threshold for Optimization
len_th=(2^r)-1;

% Pmf Vector
g_1=zeros(1,len_th+1);
g_0=zeros(1,len_th+1);

% The Vector for function
y=zeros(1,len_th);

% The Vector for function
z_1=zeros(1,len_th);
z_2=zeros(1,len_th);

% The Vector for function
F=zeros(1,len_th);

% Loop for the individual Pmf terms
for i=1:len_th+1
    
    % The Pmf corresponding to the first Threshold
    if(i==1)
        
        g_1(i)=normcdf((th(1)-m_a)/sqrt(var_s));
        g_0(i)=normcdf(th(1)/sqrt(var_s));
        
        %  The Pmf for the Last Threshold
    elseif(i==len_th+1)
        
        g_1(i)=1-normcdf((th(end)-m_a)/sqrt(var_s));
        g_0(i)=1-normcdf(th(end)/sqrt(var_s));
        
        % The Pmf for other Threshold
    else
        
        g_1(i)=normcdf((th(i)-m_a)/sqrt(var_s))-normcdf((th(i-1)-m_a)/sqrt(var_s));
        g_0(i)=normcdf(th(i)/sqrt(var_s))-normcdf(th(i-1)/sqrt(var_s));
        
    end
    
end
    
    % Loop for the Individual function terms
    for j=1:len_th
        
         % The First Term
        y(j)=exp((-0.5/var_s)*((m_a^2)-(2*m_a*th(j))));
        
        % The Second Term
        z_1(j)=((g_1(j)/g_0(j))-(g_1(j+1)/g_0(j+1)));
        
        % The Third Term
        z_2(j)=log((g_1(j)/g_0(j))/(g_1(j+1)/g_0(j+1)));
        
        % The function
        F(j)=y(j)-(z_1(j)/z_2(j));
        
    end
    
end
