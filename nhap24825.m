clc; clear; close all;

% --- Tham số mô phỏng ---
N = 200; 
x = linspace(-1,1,N);
[X,Y] = meshgrid(x,x);

% --- Pha ground truth (object phase) ---
phi_true = 5*X + 3*Y + 2*sin(5*X).*cos(5*Y);  

% --- Thêm noise vào pha thật trước khi wrap ---
noise_level = 0.5;  % mức nhiễu rad
phi_noisy = phi_true + noise_level*randn(size(phi_true));

% --- Sinh wrapped phase ---
phi_wrap = angle(exp(1i*phi_noisy));

% --- Sinh estimate (ví dụ: ground truth + nhiễu nhẹ) ---
phi_est = phi_true + 0.2*randn(size(phi_true));

% --- Unwrap theo TIE-DCT gốc ---
[phi_unwrap_iter, N_iter] = Unwrap_TIE_DCT_Iter(phi_wrap);

% --- Unwrap theo TIE-DCT có estimate ---
[phi_unwrap_est, N_est] = Unwrap_TIE_DCT_Iter_Est(phi_wrap, phi_est);

% --- Đánh giá sai số ---
err_iter = mean(abs(phi_true(:)-phi_unwrap_iter(:)));
err_est  = mean(abs(phi_true(:)-phi_unwrap_est(:)));

fprintf('Sai số trung bình (TIE-DCT): %.4f rad\n', err_iter);
fprintf('Sai số trung bình (TIE-DCT + estimate): %.4f rad\n', err_est);

% --- Hiển thị kết quả ---
figure;
subplot(2,3,1); imagesc(phi_true); axis image; colorbar; title('Ground Truth Phase');
subplot(2,3,2); imagesc(phi_noisy); axis image; colorbar; title('Noisy True Phase');
subplot(2,3,3); imagesc(phi_wrap); axis image; colorbar; title('Wrapped Phase');
subplot(2,3,4); imagesc(phi_unwrap_iter); axis image; colorbar; 
title(['Unwrap TIE-DCT, N=',num2str(N_iter)]);
subplot(2,3,5); imagesc(phi_unwrap_est); axis image; colorbar; 
title(['Unwrap TIE-DCT + Estimate, N=',num2str(N_est)]);
subplot(2,3,6); imagesc(phi_true - phi_unwrap_est); axis image; colorbar; 
title('Error map (Est-based)');





function [phase_unwrap,N]=Unwrap_TIE_DCT_Iter_Est(phase_wrap, phi_est)
   % --- khởi tạo từ estimate ---
   K = round((phi_est - phase_wrap)/(2*pi));
   phase_unwrap = phase_wrap + 2*pi*K;
   phi = phi_est; 
   N = 0;
   
   % --- refine iterative ---
   residue = wrapToPi(phase_unwrap - phi);
   while max(abs(residue(:))) > 1e-3
       phi = phi + unwrap_TIE(residue);
       phi = phi + mean2(phase_wrap) - mean2(phi); % adjust piston
       K = round((phi - phase_wrap)/(2*pi));
       phase_unwrap = phase_wrap + 2*pi*K;
       residue = wrapToPi(phase_unwrap - phi);
       N = N + 1;
   end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D phase Unwrapping algorithm based on a manuscript entitled "Robust 2D phase unwrapping algorithm based on the transport of intensity equation",which was submitted to Measurement Science and Technology(MST).
% Inputs:
%   * phase_wrap: wrapped phase from -pi to pi
% Output:
%   * phase_unwrap: unwrapped phase 
%   * N: number of iterations 
% Author:Zixin Zhao (Xi'an Jiaotong University, 08-15-2018)
% Email:zixinzhao@xjtu.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [phase_unwrap,N]=Unwrap_TIE_DCT_Iter(phase_wrap)   
   phi1 = unwrap_TIE(phase_wrap);
   phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
    K1=round((phi1-phase_wrap)/2/pi);  %calculate integer K
    phase_unwrap=phase_wrap+2*K1*pi; 
    residue=wrapToPi(phase_unwrap-phi1);
    phi1=phi1+unwrap_TIE(residue);
    phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
    K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
    phase_unwrap=phase_wrap+2*K2*pi; 
    residue=wrapToPi(phase_unwrap-phi1);
    N=0;
   while sum(sum(abs(K2-K1)))>0 
       K1=K2;
       phic=unwrap_TIE(residue);
     phi1=phi1+phic;
     phi1=phi1+mean2(phase_wrap)-mean2(phi1); %adjust piston
    K2=round((phi1-phase_wrap)/2/pi);  %calculate integer K
    phase_unwrap=phase_wrap+2*K2*pi; 
    residue=wrapToPi(phase_unwrap-phi1);
    N=N+1;
   end
end
function [phase_unwrap]=unwrap_TIE(phase_wrap)
      psi=exp(1i*phase_wrap);
      edx = [zeros([size(psi,1),1]), wrapToPi(diff(psi, 1, 2)), zeros([size(psi,1),1])];
      edy = [zeros([1,size(psi,2)]); wrapToPi(diff(psi, 1, 1)); zeros([1,size(psi,2)])];
       lap = diff(edx, 1, 2) + diff(edy, 1, 1); %calculate Laplacian using the finite difference
        rho=imag(conj(psi).*lap);   % calculate right hand side of Eq.(4) in the manuscript
   phase_unwrap = solvePoisson(rho); 
end
function phi = solvePoisson(rho)
    % solve the poisson equation using DCT
    dctRho = dct2(rho);
    [N, M] = size(rho);
    [I, J] = meshgrid(0:M-1, 0:N-1);
    dctPhi = dctRho ./ 2 ./ (cos(pi*I/M) + cos(pi*J/N) - 2);
    dctPhi(1,1) = 0; % handling the inf/nan value
    % now invert to get the result
    phi = idct2(dctPhi);
end
