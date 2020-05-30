function population=getPopulation(class, input,nPop)
[center,U] = fcm(input, 3, [2 100 1e-5]); %center = center cluster, U = membership level
[~, Hi] = max(U); % Yy = max value between both membership function, Hi = the class corresponding to the max value
[~, features] = size(input);
population = zeros(nPop, 3, class, features); % parameter: population size * 6 * total classes * total features
for loop=1:nPop
    a = zeros(class, features);
    b = repmat(2, class, features);
    c = zeros(class, features);
            for k = 1:class
                for i = 1:features % looping for all features
                    % premise parameter: a
                    
                    Ri=max(input(:, i))-min(input(:, i));
                    r=sum(Hi' == k);
                    aTemp = (Ri)/(2*r-2);
                    aLower = aTemp*0.5;
                    aUpper = aTemp*1.5;
                    a(k, i) = (aUpper-aLower).*rand()+aLower;

                    %premise parameter: c
                    dcc = (2.1-1.9).*rand()+1.9;
                    cLower = center(k,features)-dcc/2;
                    cUpper = center(k,features)+dcc/2;
                    c(k,i) = (cUpper-cLower).*rand()+cLower;
                end
            end
            population(loop, 1, :, :) = a;
            population(loop, 2, :, :) = b;
            population(loop, 3, :, :) = c;
end
end