function[I]=Lauricella_A(p)

% The Lauricella A function computation

% Constant Coefficient
d=(gamma((p+2)/2))^-1;

% Integrand Function
    function y=fun(t)

        % Intergrand term
        y=exp((-1).*t).*(t.^(((p+2)/2)-1)).*((hypergeom(0.5,1.5,(-1).*t)).^p);
        
    end

% Function Handle
f=@fun;

% Integration
q=integral(f,0,Inf);

% Value of Lauricella A function
I=d*q;

end

