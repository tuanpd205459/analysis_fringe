% 2D phase Unwrapping algorithm based on direct transport of intensity equation(TIE) method using the fast Fourier transform(FFTs)
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
% Author:Zixin Zhao (Xi'an Jiaotong University, 06-10-2019)
% Email:zixinzhao@xjtu.edu.cn
%references:
%1.M.A. Schofield, Y. Zhu, Fast phase unwrapping algorithm for interferometric applications, Optics letters, 28 (2003) 1194-1196.
%2.J. Martinez-Carranza, K. Falaggis, T. Kozacki, Fast and accurate phase-unwrapping algorithm based on the transport of intensity equation, Appl Opt, 56 (2017) 7079-7088.
%3.https://ww2.mathworks.cn/matlabcentral/fileexchange/67327-simple-transport-of-intensity-equation-phase-unwrapper-stiepu-m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [phase_unwrap]=unwrap_TIE_FFT_FFT(phase_wrap)
[M,N]=size(phase_wrap);
p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
[height,width]=size(p1);
mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle 
fx=2*pi*1i*mx/width;
fy=2*pi*1i*my/height;
[fx,fy]=meshgrid(fx,fy);
f=fx.^2+fy.^2;
u0=exp(1i*p1);
temp1=fftshift(fft2(u0));
temp1=temp1.*f;
temp2=ifft2(ifftshift(temp1));
temp2=imag(temp2.*exp(-1i*p1));%temp2 is rho;
temp3=fftshift(fft2(temp2));
temp3=temp3./(f+eps);
p2=ifft2(ifftshift(temp3));
phase_unwrap=p2(1:M,1:N);
end