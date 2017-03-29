function results=run_MTA(seq, res_path, bSaveImage)

close all;

x=seq.init_rect(1)-1;%matlab to c
y=seq.init_rect(2)-1;
w=seq.init_rect(3);
h=seq.init_rect(4);

%featureName kernelName param svmC svmBudgetSize searchRadius seed
%featureName: raw haar histogram
%kernelName: linear gaussian intersection chi2
%seed: default - 0
tic
command = ['MTA.exe' ' ' seq.name ' ' seq.path ' ' num2str(seq.startFrame) ' ' num2str(seq.endFrame) ' ' num2str(x) ' ' num2str(y) ' ' num2str(w) ' ' num2str(h) ' '  num2str(seq.nz) ' ' seq.ext ' ' '0'];
dos(command);
duration=toc;

results.res = dlmread([seq.name '_MRT.txt']);
results.res(:,1:2) =results.res(:,1:2) + 1;%c to matlab
delete([seq.name '_MRT.txt'])

results.type='rect';
results.fps=seq.len/duration;