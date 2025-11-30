% my_zernike
% hard-code 36 he so dau Zernike

clc, clear, close all;
%%
gridsize = 512;
x = linspace(-1,1, gridsize);
y = linspace(-1,1, gridsize);
[X, Y] = meshgrid(x,y);

[theta, rho] = cart2pol(X,Y);
% TẠO MASK: Chỉ lấy giá trị trong hình tròn bán kính 1
mask = rho <= 1;

% Mảng 1: Vị trí (Index) các Zernike 
% pos_coeff = [1, 2, 3, 4, 5, 6, 7 ]; 
% val_coeff = [];

% --- 2. Chọn đa thức Zernike và tạo hệ số ngẫu nhiên ---
pos_coeff = 4:10;      % index các đa thức Zernike
max_amp = 3;
val_coeff = (rand(1, length(pos_coeff)) - 0.5) * 2 * max_amp;
for k = 1:length(pos_coeff)
    fprintf('Index: %d, Coeff: %.2f\n', pos_coeff(k), val_coeff(k));
end

% Kiểm tra an toàn: Hai mảng phải có cùng độ dài
if length(pos_coeff) ~= length(val_coeff)
    error('Lỗi: Số lượng vị trí và số lượng hệ số không bằng nhau!');
end
surface = zeros(size(rho));
for k = 1:length(pos_coeff)
    j = pos_coeff(k); 
    c = val_coeff(k);  
    if c==0
        continue;
    end
    Z_term = my_get_zernike_poly(j, rho, theta);
    surface = surface + c*Z_term;
end
%%
surface_HCN = surface;
% Áp dụng Mask (Gán NaN cho vùng ngoài hình tròn để vẽ đẹp hơn)
surface(~mask) = NaN;

figure;
surf(surface,"EdgeColor","none");
title('anh tai tao Zernike');

figure;
surf(surface_HCN,"EdgeColor","none");
title('anh tai tao Zernike - HCN');
colormap turbo; colorbar;
%% fitting bề mặt
save("my_create_zernike.mat");