% 2D phase Unwrapping algorithm based on direct transport of intensity equation(TIE) method using the finite difference(FD) and fast Fourier transform(FFT)
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
% Author:Zixin Zhao (Xi'an Jiaotong University, 06-10-2019)
% Email:zixinzhao@xjtu.edu.cn
%references:
%1.Z. Zhao, H. Zhang,etc,Phase unwrapping algorithms based on solving the Poisson equation£ºA comparative review, submitted to Optics and Lasers in Engineering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [phase_unwrap]=unwrap_TIE_FD_FFT(phase_wrap)
[M,N]=size(phase_wrap);
p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
[height,width]=size(p1);
mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle 
fx=2*pi*1i*mx/width;
fy=2*pi*1i*my/height;
[fx,fy]=meshgrid(fx,fy);
f=fx.^2+fy.^2;
 psi=exp(1i*p1);
 edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
 edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
  lap = diff(edx, 1, 2) + diff(edy, 1, 1);
   rho=imag(conj(psi).*lap);
temp2=rho;
temp3=fftshift(fft2(temp2));
temp3=temp3./(f+eps);
p2=ifft2(ifftshift(temp3));
phase_unwrap=p2(1:M,1:N);
end