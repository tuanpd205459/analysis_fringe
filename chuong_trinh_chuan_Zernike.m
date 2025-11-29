clc, clear, close all;
% chương trình chính dùng để loại bỏ nghiêng bề mặt bằng Zernike
% tuwf anh thuc
tic
%%
load("chuong_trinh_chinh_anh_that.mat");
figure;
surf(finalUnwrappedPhase,"EdgeColor","none");
title("anh proposal");

%%
z_map = finalUnwrappedPhase;
coeff = zeros(1, 2);
coeff(1) = 25; coeff(2) = 25;
[output_coeff, z_recon_map2] = ZernikeLegendreFit_removal(z_map, "2indices", coeff);

error_removal = z_recon_map2 - z_map;
figure;
surf(error_removal,"EdgeColor","none");
title("sai so sau khi removal giua fitting va gt");

%% làm mượt đỉnh sin
% z_recon_map2 = imgaussfilt(z_recon_map2, [10 10]); % cách 1: dùng bộ lọc
% cách 2: 
%  sigma = 2;         % bạn chỉnh 1–5 tùy mức mượt
% window = 6*sigma;  % nên >= 6*sigma
% h = fspecial('gaussian', window, sigma);
% 
% z_recon_map2 = imfilter(z_recon_map2, h, 'replicate');

% % cách 3:
% [m, n] = size(z_recon_map2);
% Z = fft2(z_recon_map2);
% Z_shift = fftshift(Z);
% 
% % tạo mặt nạ lọc low-pass
% cutoff = 0.1; % 0–0.5, nhỏ hơn = mượt hơn
% [X, Y] = meshgrid(linspace(-1,1,n), linspace(-1,1,m));
% mask = sqrt(X.^2 + Y.^2) < cutoff;
% 
% Z_filtered = Z_shift .* mask;
% z_recon_map2 = real(ifft2(ifftshift(Z_filtered)));
% 
% figure; surf(z_recon_map2); shading interp; title('FFT low-pass smoothing');

%%
figure;
surf(z_recon_map2, "EdgeColor","none");
title('ảnh dùng Zernike Legendre abberation removal');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();

toc
