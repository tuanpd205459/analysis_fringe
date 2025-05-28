clc; clear; close all;

%% ==== Thông số cơ bản ====
N = 1080;                  % Kích thước ảnh
lambda = 632.8e-9;         % Bước sóng ánh sáng (m)
k = 2 * pi / lambda;       % Số sóng
[x, y] = meshgrid(linspace(-1, 1, N));  % Lưới tọa độ chuẩn hóa

%% ==== Sóng vật với mặt cong + bề mặt Gaussian ====
R_obj = 5e3;               % Bán kính cong sóng vật (m)
ampPhase = 4;              % Biên độ pha cho profile vật

% Pha sóng vật = mặt sóng cầu + profile Gaussian
phi_surface = ampPhase * exp(-10 * (x.^2 + y.^2));
phi_obj = k * ((x * N/2).^2 + (y * N/2).^2) / (2 * R_obj) + phi_surface;
phi_obj = 1e-7 * phi_obj;
% Sóng vật
Es = exp(1i * phi_obj);

%% ==== Sóng tham chiếu lệch trục ====
theta = 5 * pi / 180;      % Góc lệch trục (rad)
[Xa, ~] = meshgrid(1:N, 1:N);
phi_ref = 1e-7 * k * sin(theta) * Xa;
E0 = exp(1i * phi_ref);    % Sóng tham chiếu

%% ==== Giao thoa và chuyển pha về chiều cao ====
I = abs(E0 + Es).^2;                           % Ảnh giao thoa
h_surface = (lambda / (4 * pi)) * phi_obj;     % Độ cao bề mặt (m)
%%
% 3D surface plot of the object
figure('Name', 'True Object Height Map');
surf(h_surface, 'EdgeColor', 'none');
colormap turbo;
xlabel('x (px)'); ylabel('y (px)'); zlabel('Height (m)');
title('Object Surface Height (True, from Phase)');
colorbar;
view([45 30]);
c = colorbar; 
c.Label.String = 'Height (m)';

%
figure;
imagesc(I);
title("ảnh giao thoa");
figure;
imagesc(angle(E0)); axis square; colormap(jet); colorbar; axis off;
title('Reference Wave Phase (\phi_{ref})');
%% ==== Lưu kết quả ====
save("ket_qua_mo_phong.mat", "h_surface", "I");
