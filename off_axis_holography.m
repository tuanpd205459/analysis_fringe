%% Mô phỏng giao thoa off-axis holography đơn giản

clc; clear; close all;

%% Kích thước ảnh (số điểm ảnh CCD)
Ax = 256;     
Ay = 256;

%% Lưới tọa độ không gian
[Xa, Ya] = meshgrid(1:Ax, 1:Ay);

%% Tạo sóng vật thể Es với mặt pha parabol
% phi_vat = 0.001 * ((Xa - Ax/2).^2 + (Ya - Ay/2).^2); 

N = 256;
ampPhase = 10;
noise = 0;
[x, y] = meshgrid(linspace(-1,1,N));
%%%%% (1) unweighted case
% original unwrapped phase
% phi_vat = exp(-(x.*x+y.*y)/2/0.2^2) * ampPhase + (x + y) * ampPhase/2;
phi_vat = ampPhase * exp(-10*(x.^2 + y.^2));
%%
% N = 256;
% [x,y]=meshgrid(1:N);
% phi_vat = 2*peaks(N) + 0.1*x + 0.01*y;
% Thêm 1 đỉnh lồi tại vị trí A, 1 đỉnh lõm tại vị trí B

%%
Es = exp(1i * phi_vat);  % Trường sóng vật thể

%% Hiển thị mặt pha và biên độ sóng vật thể
figure;
subplot(1,2,1)
imagesc(angle(Es)); 
title('Pha sóng vật thể'); 
axis square; colormap(hsv); colorbar; axis off;

subplot(1,2,2)
surf(phi_vat, 'EdgeColor', 'none');
title('Bề mặt sóng vật thể'); 
xlabel('x'); ylabel('y'); zlabel('\phi');
colormap(jet); colorbar; view([45 30]);

%% Tạo sóng tham chiếu nghiêng theo trục x (off-axis)
lambda = 1;                              
theta = 5 * pi / 180;                   
k = 2 * pi / lambda;                    
kSinTheta = k * sin(theta);            
phi_tc = kSinTheta * Xa;
E0 = exp(1i * phi_tc);  % Sóng tham chiếu

%% Hiển thị pha sóng tham chiếu và bề mặt
figure;
subplot(1,2,1)
imagesc(angle(E0)); 
title('Pha sóng tham chiếu'); 
axis square; colormap(jet); colorbar; axis off;

subplot(1,2,2)
surf(phi_tc, 'EdgeColor', 'none'); 
title('Bề mặt sóng tham chiếu'); 
xlabel('x'); ylabel('y'); zlabel('\phi_{ref}');
colormap(jet); colorbar; view([45 30]);

%% Mô phỏng ảnh giao thoa
I = abs(E0 + Es).^2;

%% Hiển thị ảnh giao thoa
figure;
imagesc(I); 
title('Ảnh giao thoa off-axis'); 
colormap(gray); axis square; axis off;

%% Biến đổi Fourier và lọc để tái thiết sóng vật thể
nb = 3;  
Fh = fftshift(fft2(I, nb*Ax, nb*Ay));

Sfreq = (-1/2:1/(nb*Ax):1/2-1/(nb*Ax));
[Sx, Sy] = meshgrid(Sfreq, Sfreq);

%% Hiển thị phổ Fourier
figure;
imagesc(Sfreq, Sfreq, abs(Fh)); 
axis square; title('Phổ Fourier của ảnh giao thoa'); 
colormap(jet); colorbar;

%% Lọc bậc -1 và dịch về giữa
freq = kSinTheta / (2*pi);    
width = freq;                

Mask1 = (Sx > -freq-width/2) & (Sx < -freq+width/2);
Fh2 = Fh .* Mask1;

figure;
imagesc(Sfreq, Sfreq, abs(Fh2)); 
axis square; title('Lọc bậc -1 trong miền Fourier');
colormap(jet); colorbar;

% Dịch phổ về giữa
Mask2 = (Sx > -width/2) & (Sx < width/2);  
Fh3 = zeros(size(Fh));
Fh3(Mask2) = Fh2(Mask1);

figure;
imagesc(Sfreq, Sfreq, abs(Fh3)); 
axis square; title('Phổ sau khi dịch về giữa');
colormap(jet); colorbar;

%% Biến đổi ngược để tái thiết sóng vật thể
tempIFT = ifft2(ifftshift(Fh3));
finalField = tempIFT(1:Ax, 1:Ay);

%% So sánh pha ban đầu và pha tái thiết
figure;
subplot(1,2,1)
imagesc(angle(Es)); 
title('Pha ban đầu'); 
axis square; colormap(jet); clim([-pi pi]); colorbar; axis off;

subplot(1,2,2)
imagesc(angle(finalField)); 
title('Pha tái thiết'); 
axis square; colormap(jet); clim([-pi pi]); colorbar; axis off;
%%
%% Unwrapping pha và so sánh

% Pha ban đầu (tham khảo) và tái tạo
phi_true = angle(Es);
phi_recon = angle(finalField);

% Hiển thị kết quả
figure;
surf(phi_recon, 'EdgeColor', 'none'); 
title(' wrapped phase');
axis square; colormap(jet); colorbar; 

% Unwrap theo cả hai chiều
% phi_true_unwrapped = unwrap(unwrap(phi_true, [], 1), [], 2);
phi_recon_unwrapped = Unwrap_TIE_DCT_Iter(phi_recon);
phi_true_unwrapped = phi_vat;
% Hiển thị kết quả
figure;
surf(phi_true_unwrapped, 'EdgeColor', 'none'); 
title('Pha ban đầu (unwrapped)');
axis square; colormap(jet); colorbar; 

figure;
surf(phi_recon_unwrapped, 'EdgeColor', 'none'); 
title('Pha sau tái tạo (unwrapped)');
colormap(jet); colorbar; view([45 30]);

figure;
surf(phi_recon_unwrapped - phi_true_unwrapped, 'EdgeColor', 'none'); 
title('Sai lệch sau tái tạo');
colormap(jet); colorbar; view([45 30]);


%% tính K từ thuật toán tái tạo LS

wrapped_phase_est = atan2(sin(phi_recon_unwrapped),cos(phi_recon_unwrapped));
k_est = round((phi_recon_unwrapped - wrapped_phase_est) / (2*pi));
% k_est(isnan(k_est)) = 0;

figure;
surf(k_est, 'EdgeColor', 'none'); 
title('k est');
colormap(jet); colorbar; view([45 30]);


%%
phi_refined = phi_recon + k_est *2*pi;
figure;
surf(phi_refined, 'EdgeColor', 'none'); 
title('Pha tais tao bang k-est');
colormap(jet); colorbar; view([45 30]);