
tic;
clc;
clear;
close all;

% load dataset
data = csvread('routput.csv',1,0);
input = data(:, 1:end-1);
target = data(:, end);


% Initialize parameter
[center,U] = fcm(input, 3, [2 100 1e-3]); %center = center cluster, U = membership level
[samples, features] = size(input);
class = 3; % total classes
ep = 0;
epMax = 1;
[Yy, Hi] = max(U); % Yy = max value between both membership function, Hi = the class corresponding to the max value

bestMSE = 1000;
besta = zeros(class, features);
bestb = repmat(2, class, features);
bestc = zeros(class, features);

while ep < epMax
    ep = ep + 1;
    a = zeros(class, features);
%     b = repmat(2, class, total_features);
b = zeros(class, features);
    c = zeros(class, features);
    % Estimating random mf parameters
    for k =1:class
        for i = 1:features % looping for all features
            % premise parameter: a
            Rj=max(input(:, i))-min(input(:, i));
            m=sum(Hi' == k);
            aTemp = (Rj)/(2*m-2);
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
    
    H = [];
    Mu = zeros(samples, class, features); % Mu: miu all samples (total samples x total classes x total features)
    %Calculate membership grades
    for i = 1:samples % looping for each samples for forward pass
        %Calculate firing strength
        for k = 1:class
            w1(k) = 1; % w (not w bar)
            for j = 1:features
                mu(k,j) = 1/(1 + ((input(i,j)-c(k,j))/a(k,j))^(2*b(k,j))); % mu: miu of one sample
                w1(k) = w1(k)*mu(k,j); % fill w for k-th class
                Mu(i, k, j) = mu(k, j);
            end;
        end;
        %Calculate Normalised Firing
        w = w1/sum(w1); % w = w bar one row / one sample data
        ZX = [];
        %Generate X of f=XZ
        for k = 1:class
            ZX = [ZX w(k)*input(i,:) w(k)];
        end; 
        H = [H; ZX]; % combine matrix H of each sample
    end

    % consequent parameter (p, q, r)
    beta = pinv(H) * target; % moore pseudo invers

    output = H * beta; % calculate output from weight
    
    error=target - output;
    MSE = mean((error).^2); % calculate MSE

    if MSE < bestMSE % update min error
        bestMSE = MSE;
        bestoutput=output;
        besta = a;
        bestb = b;
        bestc = c;
    end

    Et(ep) = bestMSE;

    disp(['Iteration ' num2str(ep) ': Best Cost = ' num2str(Et(ep))]);

end;

fis.output=bestoutput;


time = toc;
PlotResult(target, fis.output,'ELANFIS', time)
