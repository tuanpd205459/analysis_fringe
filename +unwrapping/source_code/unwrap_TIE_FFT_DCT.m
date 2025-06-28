% 2D phase Unwrapping algorithm based on direct transport of intensity equation(TIE) method using the fast Fourier transform(FFT) and discret cosine transform(DCT)
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
% Author:Zixin Zhao (Xi'an Jiaotong University, 06-10-2019)
% Email:zixinzhao@xjtu.edu.cn
%references:
%1.Z. Zhao, H. Zhang,etc,Phase unwrapping algorithms based on solving the Poisson equation£ºA comparative review, submitted to Optics and Lasers in Engineering
%2.https://ww2.mathworks.cn/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [phase_unwrap]=unwrap_TIE_FFT_DCT(phase_wrap)
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
temp2=imag(temp2.*exp(-1i*p1));%temp2 is rou;
rho=temp2(1:M,1:N);
phase_unwrap = solvePoisson(rho);
end

function phi = solvePoisson(rho)
    % solve the poisson equation using dct
    dctRho = dct2(rho);
    [N, M] = size(rho);
    [I, J] = meshgrid([0:M-1], [0:N-1]);
    dctPhi = dctRho ./ 2 ./ (cos(pi*I/M) + cos(pi*J/N) - 2);
    dctPhi(1,1) = 0; % handling the inf/nan value
    
    % now invert to get the result
    phi = idct2(dctPhi);
    
end