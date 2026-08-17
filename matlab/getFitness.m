function [MSE,output] = getFitness(premise, class, input, target)
%{
get_fitness = Calculate fitness value for each premise parameter (a, b, c) of one particle

*) Output Parameter:
    - fitness = fitness value result
*) Input Parameter:
    - premise = premise parameter, size: 3 (a, b, and c parameter) x total
    classes x total features
    - class   = total classes
    - data    = input data, total samples x total features
%}

[samples, features] = size(input);
H = [];
Mu = zeros(samples, class, features); % Mu: miu all samples (total samples x total classes x total features)
%Calculate membership grades
% forward pass
for i = 1:samples
    %Calculate firing strength
    for k = 1:class
        w1(k) = 1; % w (not w bar)
        for j = 1:features
            mu(k, j) = 1/(1 + ((input(i, j)-premise(3, k, j))/premise(1, k, j))^(2*premise(2, k, j))); % mu: miu of one sample
            w1(k) = w1(k)*mu(k, j); % fill w of k-th class
            Mu(i, k, j) = mu(k, j);
        end
    end
    %Calculate Normalised Firing
    w = w1/sum(w1); % w = w bar of one row / one sample data
    XZ = [];
    %Generate X of f=XZ
    for k = 1:class
        XZ = [XZ w(k)*input(i,:) w(k)];
    end 
    H = [H; XZ]; % combine matrix H of each sample
end
% end of forward pass

% find consequent parameter (p, q, r)
beta = pinv(H) * target; % moore pseudo invers
output = H * beta; % calculate weight to output
error=target - output;
MSE=mean((error).^2); % calculate MSE
end