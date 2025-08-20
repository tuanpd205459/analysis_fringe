clc; clear; close all;

%% ================== 1. TẠO DỮ LIỆU PHA GIẢ LẬP ==================
M = 256; N = 256;
[x,y] = meshgrid(linspace(-2,2,N), linspace(-2,2,M));

% Pha object (ground truth)
phi = pi/2 * (x.^2 + y.^2);   % ví dụ mặt parabol

% Interferogram lý tưởng
I_clean = 1 + cos(phi);

%% ================== 2. THÊM GAUSSIAN NOISE VỚI 25 dB ==================
SNR_dB = 25;
I_noisy = awgn(I_clean, SNR_dB, 'measured');  % thêm nhiễu Gaussian

%% ================== 3. HIỂN THỊ ==================
figure;
subplot(1,3,1);
imagesc(phi); axis image; colormap jet; colorbar;
title('Ground truth phase');

subplot(1,3,2);
imagesc(I_clean); axis image; colormap gray; colorbar;
title('Interferogram (clean)');

subplot(1,3,3);
imagesc(I_noisy); axis image; colormap gray; colorbar;
title(sprintf('Interferogram with %d dB Gaussian noise', SNR_dB));
