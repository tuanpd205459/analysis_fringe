clc, clear, close all;
load("chuong_trinh_chinh_anh_that.mat");

z_map = finalUnwrappedPhase;
figure;
surf(z_map, "EdgeColor","none");
title('gt');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();

%% m, n indices
tic
coeff = zeros(1, 2);
coeff(1) = 25; coeff(2) = 25;
[output_coeff, z_recon_map] = ZernikeLegendreFit(z_map, "2indices", coeff);

figure;
surf(z_recon_map, "EdgeColor","none");
title('anhr tais tao dung Zernike Legendre Fit');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();

figure;
surf(z_recon_map - z_map,"EdgeColor","none");
title("sai so giuawx be mat fitting va gt");
toc
%%
%% m, n indices

[output_coeff, z_recon_map2] = ZernikeLegendreFit_removal(z_map, "2indices", coeff);

figure;
surf(z_recon_map2, "EdgeColor","none");
title('ảnh dùng Zernike Legendre abberation removal');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();
error_removal = z_recon_map2 - z_map;
figure;
surf(error_removal,"EdgeColor","none");
title("sai so sau khi removal giua fitting va gt");

%%
figure;
surf(z_recon_map2 - z_recon_map - error_removal,"EdgeColor","none");
title("sai so giữa 2 pp - tru di sai so removal vs gt");
