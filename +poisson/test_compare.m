clear;clc;close all
N=512;
[x,y]=meshgrid(linspace(-1,1,N));
phase_origin=8*peaks(N);
sub_colum=6;
sub_row=6;
foldername=['result_',datestr(clock, 'dd-mmm-yyyy_HH-MM-SS')];
mkdir(foldername)
i=1;
k=1;
for sigma=0:0.5:1
 phase_noise=phase_origin+sigma*randn(N);
phase_wrap=wrapToPi(phase_noise);
tic
[phase_unwrap1(:,:,k)]=unwrap_LS_FD_FFT(phase_wrap);
time1(k)=toc;
tic
[phase_unwrap2(:,:,k)]=unwrap_LS_FD_DCT(phase_wrap);
time2(k)=toc;
tic
[phase_unwrap3(:,:,k)]=unwrap_TIE_FD_FFT(phase_wrap);
time3(k)=toc;
tic
[phase_unwrap4(:,:,k)]=unwrap_TIE_FD_DCT(phase_wrap);
time4(k)=toc;
tic
[phase_unwrap5(:,:,k)]=unwrap_TIE_FFT_FFT(phase_wrap);
time5(k)=toc;
tic
[phase_unwrap6(:,:,k)]=unwrap_TIE_FFT_DCT(phase_wrap);
time6(k)=toc;
error1(:,:,k)=phase_unwrap1(:,:,k)-phase_noise-mean2(phase_unwrap1(:,:,k)-phase_noise);
[~,rms1(k)]=slove_pr(error1(:,:,k));
error2(:,:,k)=phase_unwrap2(:,:,k)-phase_noise-mean2(phase_unwrap2(:,:,k)-phase_noise);
[~,rms2(k)]=slove_pr(error2(:,:,k));
error3(:,:,k)=phase_unwrap3(:,:,k)-phase_noise-mean2(phase_unwrap3(:,:,k)-phase_noise);
[~,rms3(k)]=slove_pr(error3(:,:,k));
error4(:,:,k)=phase_unwrap4(:,:,k)-phase_noise-mean2(phase_unwrap4(:,:,k)-phase_noise);
[~,rms4(k)]=slove_pr(error4(:,:,k));
error5(:,:,k)=phase_unwrap5(:,:,k)-phase_noise-mean2(phase_unwrap5(:,:,k)-phase_noise);
[~,rms5(k)]=slove_pr(error5(:,:,k));
error6(:,:,k)=phase_unwrap6(:,:,k)-phase_noise-mean2(phase_unwrap6(:,:,k)-phase_noise);
[~,rms6(k)]=slove_pr(error6(:,:,k));
subplot(sub_row,sub_colum,1+sub_colum*(i-1)),my_display( phase_unwrap1(:,:,k) ),title(['(a' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,1+sub_colum*i),my_display( error1(:,:,k) ),title(['(a' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,2+sub_colum*(i-1)),my_display( phase_unwrap2(:,:,k) ),title(['(b' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,2+sub_colum*i),my_display( error2(:,:,k) ),title(['(b' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,3+sub_colum*(i-1)),my_display( phase_unwrap3(:,:,k) ),title(['(c' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,3+sub_colum*i),my_display(error3(:,:,k) ),title(['(c' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,4+sub_colum*(i-1)),my_display( phase_unwrap4(:,:,k) ),title(['(d' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,4+sub_colum*i),my_display( error4(:,:,k) ),title(['(d' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,5+sub_colum*(i-1)),my_display( phase_unwrap5(:,:,k) ),title(['(e' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,5+sub_colum*i),my_display( error5(:,:,k) ),title(['(e' num2str(i+1) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,6+sub_colum*(i-1)),my_display( phase_unwrap6(:,:,k) ),title(['(f' num2str(i) ')'],'Fontname','Times New Roman')
subplot(sub_row,sub_colum,6+sub_colum*i),my_display( error6(:,:,k) ) ,title(['(f' num2str(i+1) ')'],'Fontname','Times New Roman')
i=i+2;
k=k+1;
end
set(gcf,'outerposition',get(0,'screensize')); %maximum the figure
saveas(gcf,[foldername,'/subplot','-result.png']);
   time=[time1',time2',time3',time4',time5',time6'];
  rms=[rms1',rms2',rms3',rms4',rms5',rms6'];
  data=[time;rms];
  result={phase_unwrap1,phase_unwrap2,phase_unwrap3,phase_unwrap4,phase_unwrap5,phase_unwrap6};
  error={error1,error2,error3,error4,error5,error6};
   save([foldername,'/','Time&RMS.mat'],'data');
   save([foldername,'/','result.mat'],'result');
   save([foldername,'/','error.mat'],'error');     