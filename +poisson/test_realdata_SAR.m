clear;clc;close all
data_name={'complex_wphase_152_458'};
sub_colum=3;
sub_row=length(data_name)*4;
foldername=['result_realdata_SAR_',datestr(clock, 'dd-mmm-yyyy_HH-MM-SS')];
mkdir(foldername)
i=1;
for k=1:length(data_name)
data=['data/' data_name{k}];
load(data,'pw');
mask=~isnan(pw);
pw(isnan(pw))=0;
phase_wrap=pw;
tic
[phase_unwrap1(:,:,k)]=unwrap_LS_FD_FFT(phase_wrap);
time1(k)=toc;
tic
[phase_unwrap1_iter(:,:,k)]=unwrap_LS_FD_FFT_iter(phase_wrap);
time1_iter(k)=toc;
tic
[phase_unwrap2(:,:,k)]=unwrap_LS_FD_DCT(phase_wrap);
time2(k)=toc;
tic
[phase_unwrap2_iter(:,:,k)]=unwrap_LS_FD_DCT_iter(phase_wrap);
time2_iter(k)=toc;
tic
[phase_unwrap3(:,:,k)]=unwrap_TIE_FD_FFT(phase_wrap);
time3(k)=toc;
tic
[phase_unwrap3_iter(:,:,k)]=unwrap_TIE_FD_FFT_iter(phase_wrap);
time3_iter(k)=toc;
tic
[phase_unwrap4(:,:,k)]=unwrap_TIE_FD_DCT(phase_wrap);
time4(k)=toc;
tic
[phase_unwrap4_iter(:,:,k)]=unwrap_TIE_FD_DCT_iter(phase_wrap);
time4_iter(k)=toc;
tic
[phase_unwrap5(:,:,k)]=unwrap_TIE_FFT_FFT(phase_wrap);
time5(k)=toc;
tic
[phase_unwrap5_iter(:,:,k)]=unwrap_TIE_FFT_FFT_iter(phase_wrap);
time5_iter(k)=toc;
tic
[phase_unwrap6(:,:,k)]=unwrap_TIE_FFT_DCT(phase_wrap);
time6(k)=toc;
tic
[phase_unwrap6_iter(:,:,k)]=unwrap_TIE_FFT_DCT_iter(phase_wrap);
time6_iter(k)=toc;

subplot(sub_row,sub_colum,1),my_display_3D( phase_unwrap1(:,:,k) ),title(['(a' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,4),my_display_3D( phase_unwrap1_iter(:,:,k) ),title(['(a' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,2),my_display_3D( phase_unwrap2(:,:,k) ),title(['(b' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,5),my_display_3D( phase_unwrap2_iter(:,:,k) ),title(['(b' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,3),my_display_3D( phase_unwrap3(:,:,k) ),title(['(c' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,6),my_display_3D(phase_unwrap3_iter(:,:,k) ),title(['(c' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,7),my_display_3D( phase_unwrap4(:,:,k) ),title(['(d' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,10),my_display_3D( phase_unwrap4_iter(:,:,k) ),title(['(d' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,8),my_display_3D( phase_unwrap5(:,:,k) ),title(['(e' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,11),my_display_3D( phase_unwrap5_iter(:,:,k) ),title(['(e' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,9),my_display_3D( phase_unwrap6(:,:,k) ),title(['(f' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,12),my_display_3D( phase_unwrap6_iter(:,:,k) ) ,title(['(f' num2str(i+1) ')'],'Fontname','Times New Roman')
i=i+2;
%calaulate the RMS error using the TIE-FD-DCT_iter as the benchmark
benchmark=phase_unwrap2_iter(:,:,k);
[~,RMS1(k)]=slove_pr(benchmark-phase_unwrap1(:,:,k)-mean2(benchmark-phase_unwrap1(:,:,k)));
[~,RMS1_iter(k)]=slove_pr(benchmark-phase_unwrap1_iter(:,:,k)-mean2(benchmark-phase_unwrap1_iter(:,:,k)));
[~,RMS2(k)]=slove_pr(benchmark-phase_unwrap2(:,:,k)-mean2(benchmark-phase_unwrap2(:,:,k)));
[~,RMS2_iter(k)]=slove_pr(benchmark-phase_unwrap2_iter(:,:,k)-mean2(benchmark-phase_unwrap2_iter(:,:,k)));
[~,RMS3(k)]=slove_pr(benchmark-phase_unwrap3(:,:,k)-mean2(benchmark-phase_unwrap3(:,:,k)));
[~,RMS3_iter(k)]=slove_pr(benchmark-phase_unwrap3_iter(:,:,k)-mean2(benchmark-phase_unwrap3_iter(:,:,k)));
[~,RMS4(k)]=slove_pr(benchmark-phase_unwrap4(:,:,k)-mean2(benchmark-phase_unwrap4(:,:,k)));
[~,RMS4_iter(k)]=slove_pr(benchmark-phase_unwrap4_iter(:,:,k)-mean2(benchmark-phase_unwrap4_iter(:,:,k)));
[~,RMS5(k)]=slove_pr(benchmark-phase_unwrap5(:,:,k)-mean2(benchmark-phase_unwrap5(:,:,k)));
[~,RMS5_iter(k)]=slove_pr(benchmark-phase_unwrap5_iter(:,:,k)-mean2(benchmark-phase_unwrap5_iter(:,:,k)));
[~,RMS6(k)]=slove_pr(benchmark-phase_unwrap6(:,:,k)-mean2(benchmark-phase_unwrap6(:,:,k)));
[~,RMS6_iter(k)]=slove_pr(benchmark-phase_unwrap6_iter(:,:,k)-mean2(benchmark-phase_unwrap6_iter(:,:,k)));
end
set(gcf,'outerposition',get(0,'screensize')); %maximum the figure
saveas(gcf,[foldername,'/subplot','-result.png']);  