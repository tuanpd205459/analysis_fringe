%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D phase Unwrapping algorithm based on direct least square(LS) method using the finite difference(FD) and fast Fourier transform(FFT)
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
% Author:Zixin Zhao (Xi'an Jiaotong University, 06-10-2019)
% Email:zixinzhao@xjtu.edu.cn
%references:
%1.M.D. Pritt, J.S. Shipman, Least-squares two-dimensional phase unwrapping using FFT's, IEEE Transactions on geoscience and remote sensing, 32 (1994) 706-708.
%2.Z. Zhao, H. Zhang,etc,Phase unwrapping algorithms based on solving the Poisson equation£ºA comparative review, submitted to Optics and Lasers in Engineering
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [phase_unwrap]=unwrap_LS_FD_FFT(phase_wrap)
[M,N]=size(phase_wrap);
p1=[phase_wrap,flip(phase_wrap,2);flip(phase_wrap,1),flip(flip(phase_wrap,2),1)]; %mirror padded
[height,width]=size(p1);
mx=-width/2:width/2-1;my=-height/2:height/2-1;%samle 
fx=2*pi*1i*mx/width;
fy=2*pi*1i*my/height;
[fx,fy]=meshgrid(fx,fy);
f=fx.^2+fy.^2;
psi=p1;
edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
rho = diff(edx, 1, 2) + diff(edy, 1, 1);
temp3=fftshift(fft2(rho));
temp3=temp3./(f+eps);
p2=ifft2(ifftshift(temp3));
p2=real(p2);
phase_unwrap=p2(1:M,1:N);
end