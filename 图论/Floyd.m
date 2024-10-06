function [stdt,path] = Floyd(W)
[n,m] = size(W);  
stdt = W;
path = zeros(n);
for j = 1:n
    path(:,j) = j;   
end
for i = 1:n
    path(i,i) = -1; 
end
for t=1:n   
   for i=1:n     
      for j=1:n    
          if stdt(i,t)+stdt(t,j) < stdt(i,j)
             stdt(i,j) = stdt(i,t) + stdt(t,j); 
             path(i,j) = path(i,t);   
          end
      end
   end
end
end