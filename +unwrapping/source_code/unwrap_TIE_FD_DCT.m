%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D phase Unwrapping algorithm based on direct transport of intensity equation(TIE) method using the finite difference(FD) and discret cosine transform(DCT)
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
% Author:Zixin Zhao (Xi'an Jiaotong University, 06-10-2019)
% Email:zixinzhao@xjtu.edu.cn
%references:
%1.Z. Zhao, H. Zhang, etc,Robust 2D phase unwrapping algorithm based on the transport of intensity equation, Measurement Science and Technology, 30 (2018) 015201
%2.Z. Zhao, H. Zhang,etc,Phase unwrapping algorithms based on solving the Poisson equation£ºA comparative review, submitted to Optics and Lasers in Engineering
%3.https://ww2.mathworks.cn/matlabcentral/fileexchange/60345-2d-weighted-phase-unwrapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [phase_unwrap,rho]=unwrap_TIE_FD_DCT(phase_wrap)
      psi=exp(1i*phase_wrap);
      edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
      edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
       lap = diff(edx, 1, 2) + diff(edy, 1, 1);
        rho=imag(conj(psi).*lap);  
        % get the result by solving the poisson equation
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