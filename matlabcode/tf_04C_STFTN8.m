
clc;
clear;
close all;
signal = load('sig_c.mat');
signal=signal.signal;

name=['/data1/ypg/data/RF data/2016.04C/TF/STFTN8'];
mkdir(name)
for i=1:length(signal)
    D=signal(i,:)';
    [stft, t2, f2] = stftn8(D);
    stft=abs(stft);
    Name=[name,'/','STFT',num2str(i,'%06d'),'.mat'];
    save(Name,'stft');
end