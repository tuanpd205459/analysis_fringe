%% chương trình tính độ nhám bề mặt

clc, clear, close all;

load("cchuong_trinh_chinh_tao_phase.mat");
% load("cchuan_phase_comparison.mat");
phi_proposed     = finalUnwrappedPhase;                                     % Proposed / Hybrid


coeff = [25, 25]; % Hệ số Zernike

% 1. ZERNIKE REMOVAL (Loại bỏ quang sai/nghiêng)
[~, final_phi_proposed]  = ZernikeLegendreFit_removal(phi_proposed, "2indices", coeff);

% cong thuc lien he pha va chieu cao:
% h = delta(phi).lambda/4pi
lambda = 632.8;
delta_phi = final_phi_proposed;

h = delta_phi * lambda/(4*pi);

figure;
surf(h, "EdgeColor","none");
title("chieu cao be mat (nano-met)");

%%

