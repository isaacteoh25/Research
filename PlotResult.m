function PlotResult(target, output,name,time)

    figure;
%     errorSize=length(target);
    error=target-output;
    MSE=mean(error.^2);
    RMSE=sqrt(MSE);
    errorMean=mean(error);
%     MAE=mean(abs(error));
%     MAE=norm(error-errorMean)/errorSize;
    errorstd=std(error);

%     output=round(output);
    subplot(2,2,[1 2]);
    plot(target,'b');
    hold on;
    plot(output,'r');
    legend('Target','Output');
    times=round(time,2);
    title([name ', time=' num2str(times)]);
    xlabel('Sample Index');
    grid on;

    subplot(2,2,3);
    plot(error);
    legend('Error');
    title([ 'RMSE = ' num2str(RMSE) ', MSE = ' num2str(MSE)]);
    grid on;

    subplot(2,2,4);
    histfit(error, 50);
    title(['Error StD. = ' num2str(errorstd)  ', Error Mean = ' num2str(errorMean)]);
    

    % Type file to command window:
%     type(fileName)

	if ~isempty(which('plotregression'))
    figure;
    plotregression(target, output, 'Regression');
               
    set(gcf,'Toolbar','figure');
    end

    cHeader = {'Target' 'Output'}; %dummy header
%     commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
%     commaHeader = commaHeader(:)';
%     textHeader = cell2mat(commaHeader); %cHeader in text with commas
%     %write header to file
%     fileName = fullfile(pwd, 'output.csv');
    Filename = sprintf('%s.csv', name);
    fid = fopen(Filename, 'wt');
%     fprintf(fid,'%s\n',textHeader)
    fprintf(fid, 'Target, Output\n');
    % Write data.
    fprintf(fid, '%.5f, %.5f\n', [target, output]');
    fclose(fid);

end