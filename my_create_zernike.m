% my_zernike
% hard-code 36 he so dau Zernike

clc, clear, close all;
%%
gridsize = 512;
gridsize = gridsize*sqrt(2);
x = linspace(-1,1, gridsize);
y = linspace(-1,1, gridsize);
[X, Y] = meshgrid(x,y);

[theta, rho] = cart2pol(X,Y);
% TẠO MASK: Chỉ lấy giá trị trong hình tròn bán kính 1
mask = rho <= 1;

% Mảng 1: Vị trí (Index) các Zernike 
% pos_coeff = [1, 2, 3, 4, 5, 6, 7 ]; 
% val_coeff = [];

pos_coeff = 4:16;      % index các đa thức Zernike
max_amp = 3;
val_coeff = (rand(1, length(pos_coeff)) - 0.5) * 2 * max_amp;

saveFolder = fullfile(pwd, 'input_create_Zernike');
if ~exist(saveFolder, 'dir'), mkdir(saveFolder); end
filePath = fullfile(saveFolder, 'coeff_list_create.txt');
fileID = fopen(filePath, 'w');
for k = 1:length(pos_coeff)
    fprintf(fileID, 'Index: %d, Coeff: %.2f\n', pos_coeff(k), val_coeff(k));
end
fclose(fileID);

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
%%
x_square = abs(X) <= sqrt(2)/2;
y_square = abs(Y) <= sqrt(2)/2;
mask_square = x_square & y_square;

surface_square = surface;   % hình vuông từ dữ liệu gốc (không NaN)
surface_square(~mask_square) = NaN;
% Tìm tất cả vị trí không phải NaN
[row_idx, col_idx] = find(~isnan(surface_square));

% Lấy bounding box của vùng có dữ liệu
rmin = min(row_idx);
rmax = max(row_idx);
cmin = min(col_idx);
cmax = max(col_idx);

% Cắt ra vùng vuông sạch, không NaN
surface_square = surface_square(rmin:rmax, cmin:cmax);

figure;
surf(surface_square,"EdgeColor","none");
title('anh tai tao Zernike - HCN ben trong zernike');
colormap turbo; colorbar;




%% fitting bề mặt
save("my_create_zernike.mat");