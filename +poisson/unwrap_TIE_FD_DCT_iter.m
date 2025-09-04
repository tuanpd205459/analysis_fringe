%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D phase Unwrapping algorithm based on iterative transport of intensity equation(TIE) method using the finite difference(FD) and discret cosine transform(DCT)
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
%   * N: number of iterations 
% Author:Zixin Zhao (Xi'an Jiaotong University, 06-10-2019)
% Email:zixinzhao@xjtu.edu.cn
%references:
%1.Z. Zhao, H. Zhang, etc,Robust 2D phase unwrapping algorithm based on the transport of intensity equation, Measurement Science and Technology, 30 (2018) 015201
%2.Z. Zhao, H. Zhang,etc,Phase unwrapping algorithms based on solving the Poisson equation£ºA comparative review, submitted to Optics and Lasers in Engineering


function [phase_unwrap,N]=unwrap_TIE_FD_DCT_iter(phase_wrap)   
   phi1 = unwrap_TIE_FD_DCT(phase_wrap);
   phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
    K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
    phase_unwrap=phase_wrap+2*K1*pi; 
    residue=wrapToPi(phase_unwrap-phi1);
    phi1=phi1+unwrap_TIE_FD_DCT(residue);
    phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
    K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
    phase_unwrap=phase_wrap+2*K2*pi; 
    residue=wrapToPi(phase_unwrap-phi1);
    rms1=sqrt(mean2((residue).^2));
    delta_rms=rms1;
    N=0;
   while sum(sum(abs(K2-K1)))>0 && delta_rms>10^-5
       K1=K2;
       phic=unwrap_TIE_FD_DCT(residue);
     phi1=phi1+phic;
     phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
    K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
    phase_unwrap=phase_wrap+2*K2*pi; 
    residue=wrapToPi(phase_unwrap-phi1);
   rms2=sqrt(mean2((residue).^2));
    delta_rms=abs(rms2-rms1);
    rms1=rms2;
    N=N+1;
   end
end