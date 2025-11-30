%% 
clc, clear, close all;
%
% load("my_create_zernike.mat");
load("chuong_trinh_chinh_anh_that.mat")
% khai bao surface
% surface = zeros(100);
surface = finalUnwrappedPhase;
gridsize = size(surface,2);
x = linspace(-1,1,gridsize);
[X,Y] = meshgrid(x,x);
[theta, rho ] = cart2pol(X,Y);


%% 2. FITTING BỀ MẶT (QUAN TRỌNG)
is_in_circle = (rho <= 1);

b = surface(is_in_circle); 

n_terms = 36;       % so da thuc can fit

n_pixels = length(b);

% Tạo ma trận thiết kế A
A = zeros(n_pixels, n_terms);
for j = 1:n_terms
    Z_temp = my_get_zernike_poly(j, rho, theta);
    A(:, j) = Z_temp(is_in_circle); 
end

coeffs_fitted = A \ b; 
% -----------------------------------


%% tái tạo lại mặt
reconstructed_pixels = A * coeffs_fitted; 

reconstructed_map = NaN(size(surface));
reconstructed_map(is_in_circle) = reconstructed_pixels;
% -----------------------------------

figure;
surf(reconstructed_map, "EdgeColor", "none");
title('Ảnh tái tạo từ Fitting');
axis square; view(-45, 30);
colormap turbo;
figure;
surf(surface, "EdgeColor", "none");
title('Ảnh ban dau');
axis square; view(-45, 30);
colormap turbo;
% Tính sai số RMS
diff = surface - reconstructed_map;
rms_val = sqrt(nanmean(diff(:).^2));
fprintf('\nSai số RMS: %.10f\n', rms_val);

figure;
imagesc(rms_val);
title('sai so');
colorbar;
