function results=run_KCF(seq, res_path, bSaveImage)

close all;

x=seq.init_rect(1)-1;%matlab to c
y=seq.init_rect(2)-1;
w=seq.init_rect(3);
h=seq.init_rect(4);

if bSaveImage
    show='1';
else
    show='0';
end

if ispc
    exec = 'CT.exe';
else
    exec = './CT';
    setenv('LD_LIBRARY_PATH','/usr/lib/:/usr/local/lib/:/usr/lib/x86_64-linux-gnu/');
    dos('chmod +x ./CT');
end
tic
command = [exec ' ' seq.name ' ' seq.path ' ' num2str(seq.startFrame) ' ' num2str(seq.endFrame) ' '  num2str(seq.nz) ' ' seq.ext ' ' num2str(x) ' ' num2str(y) ' ' num2str(w) ' ' num2str(h) ' ' show];
dos(command);
duration=toc;

resultPath=[seq.name '_result.txt'];
results.res = dlmread(resultPath);
results.res(:,1:2) =results.res(:,1:2) + 1;%c to matlab

if exist(resultPath,'file')==2
    delete(resultPath);
end

results.type='rect';
results.fps=seq.len/duration;

