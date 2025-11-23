clear; close all
clc;
% 31/10/2024        - chạy ổn tạo và tái tạo bề mặt zernike

% x = sin(linspace(0, 6*pi, 100));
% z_map = ones(100, 1) * x;
% Kích thước lưới
zernike_coeffs = [
    2
    0
    0
    0
    0
    2.5
    -0.25
    1
    3
    -1.12
    2.1
    1.3
    -2
    -0.5
    0.8
    1.570000000
    0.00000000
    0.00000000
    0.00000000
    0.00
    0.00000000
    0.00000000
    0.00000000
    0.00000208
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0.00000000
    0
];
% filename = 'guongvat2-1.xlsx';
% sheetname = 'guongvat2-1'; % Tên sheet cụ thể
% range = 'M45:M89'; % Phạm vi dữ liệu
% % Đọc dữ liệu từ sheet cụ thể
% data = readtable(filename, 'Sheet', sheetname, 'Range', range);

% Các biến khác
order = 15; % Bậc cao nhất của Zernike
grid_size = 512; % Kích thước lưới
% zernike_coeffs = data{:,:}; % Cột chứa các hệ số Zernike




wavefront = reconstruct_wavefront(zernike_coeffs, order, grid_size);
%wavefront = atan2(sin(wavefront), cos(wavefront));

% % Loại bỏ các giá trị ngoài đường tròn đơn vị
% [X, Y] = meshgrid(linspace(-1, 1, grid_size), linspace(-1, 1, grid_size));
% R = sqrt(X.^2 + Y.^2);
% be_mat_tai_tao = wavefront ;
% % be_mat_tai_tao = wavefront;
% % wavefront(R > 1) = NaN;
% % be_mat_tai_tao(R>1) = NaN;
figure;
% Hiển thị bề mặt sóng
surf(wavefront,"EdgeColor","none");
title('ảnh mô phỏng từ đa thức Zernike');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();

z_map = wavefront;

%% m, n indices
coeff = zeros(1, 2);
coeff(1) = 10; coeff(2) = 5;
[output_coeff, z_recon_map] = ZernikeLegendreFit(z_map, "2indices", coeff);

figure;
surf(z_recon_map, "EdgeColor","none");
title('Tái tạo bề mặt sóng từ 3D');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();

figure;
surf(z_recon_map - z_map,"EdgeColor","none");
title("sai so");
%%
%% m, n indices
coeff = zeros(1, 2);
coeff(1) = 10; coeff(2) = 5;
[output_coeff, z_recon_map] = ZernikeLegendreFit_removal(z_map, "2indices", coeff);

figure;
surf(z_recon_map, "EdgeColor","none");
title('Tái tạo bề mặt sóng từ 3D');
xlabel('X');
ylabel('Y');
zlabel('Độ lệch pha');
colormap(jet);    % Áp dụng bảng màu "jet"
colorbar();

figure;
surf(z_recon_map - z_map,"EdgeColor","none");
title("sai so sau khi removal");

%% Fringe index
% coeff = 100;
% [output_coeff, z_recon_map] = ZernikeLegendreFit(z_map, "fringe", coeff);
% 
% fprintf('He so: %.1f \n', output_coeff{1});

function surface = reconstruct_wavefront(zernike_coeffs, order, grid_size)
    % Hàm tái tạo bề mặt sóng từ các hệ số đa thức Zernike
    % zernike_coeffs: Hệ số đa thức Zernike
    % order: Bậc của các đa thức Zernike
    % grid_size: Kích thước của lưới điểm trên mặt phẳng 

    % Khởi tạo mặt phẳng lưới
    [X, Y] = meshgrid(linspace(-1, 1, grid_size), linspace(-1, 1, grid_size));
    R = sqrt(X.^2 + Y.^2);
    Theta = atan2(Y, X);

    % Khởi tạo bề mặt sóng
    surface = zeros(size(X));

    % Lặp qua các bậc và hệ số để tái tạo bề mặt sóng
    index = 1;
    for n = 0:order
        for m = -n:2:n
            if index <= length(zernike_coeffs)
                % Lấy hệ số tương ứng
                Z = zernike_coeffs(index);
                
                % Tính giá trị Zernike polynomial
                Zmn = ZernikePolynomial(n, m, R, Theta);
                
                % Cộng dồn vào bề mặt sóng
              %  surface = surface + Z * Zmn;
                surface = surface + Z * Zmn;

                index = index + 1;
            end
        end
    end
end

function Zmn = ZernikePolynomial(n, m, R, Theta)
    % Hàm tính Zernike polynomial
    % n: Bậc n của Zernike
    % m: Bậc m của Zernike
    % R: Bán kính
    % Theta: Góc theta
    
    % Tính radial Zernike polynomial
    RadialPoly = zeros(size(R));
    for k = 0:floor((n-abs(m))/2)
        c = (-1)^k * factorial(n-k) / (factorial(k) * factorial((n + abs(m))/2 - k) * factorial((n - abs(m))/2 - k));
        RadialPoly = RadialPoly + c * R.^(n - 2*k);
    end
    
    % Tính Zernike polynomial
    if m >= 0
        Zmn = RadialPoly .* cos(m * Theta);
    else
        Zmn = RadialPoly .* sin(abs(m) * Theta);
    end
end

%% Tính toán đa thức Zernike
function radial = zernike_radial(r,n,m)
    % Functions required for use: elliptical_crop
%     hàm tính toán đa thức Zernike
%     Đầu vào
%         r:      bán kính
%         n:      bậc quang sai
%         m:      bậc phương vị
%     Đầu ra:
%         giá trị Đa thức Zernike

    if mod(n-m,2) == 1
        error('n-m must be even');
    end
    if n < 0 || m < 0
        error('n and m must both be positive in radial function')
    end
    if floor(n) ~= n || floor(m) ~= m
        error('n and m must both be integers')
    end
    if n == m
        radial = r.^n;
    elseif n - m == 2
        radial = n*zernike_radial(r,n,n)-(n-1)*zernike_radial(r,n-2,n-2);
    else
        H3 = (-4*((m+4)-2)*((m+4)-3)) / ((n+(m+4)-2)*(n-(m+4)+4));
        H2 = (H3*(n+(m+4))*(n-(m+4)+2)) / (4*((m+4)-1))  +  ((m+4)-2);
        H1 = ((m+4)*((m+4)-1) / 2)  -  (m+4)*H2  +  (H3*(n+(m+4)+2)*(n-(m+4))) / (8);
        radial = H1*zernike_radial(r,n,m+4) + (H2+H3 ./ r.^2).*zernike_radial(r,n,m+2);
        
        % Fill in NaN values that may have resulted from DIV/0 in prior
        % line. Evaluate these points directly (non-recursively) as they
        % are scarce if present.
        
        if sum(sum(isnan(radial))) > 0
            [row, col] = find(isnan(radial));
            c=1;
            while c<=length(row)
                x = 0;
                for k = 0:(n-m)/2
                    ((-1)^k*factorial(n-k))/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))*0^(n-2*k);
                    x = x + ((-1)^k*factorial(n-k))/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))*0^(n-2*k);
                end
                radial(row(c),col(c)) = x;
                c=c+1;
            end
        end

    end

end
function cropped_im = elliptical_crop(im,crop_frac)
    % Hàm crop ảnh tương ứng đường tròn
    
    if crop_frac < 0 || crop_frac > 1
        error('crop_frac must have value between 0 and 1')
    end

    cropped_im = im;
    center_x = (size(im,2)+1)/2;
    center_y = (size(im,1)+1)/2;
    radius_x = (size(im,2)-center_x)*crop_frac;
    radius_y = (size(im,1)-center_y)*crop_frac;

    for row = 1:size(im,1)
        for col = 1:size(im,2)
            if sqrt((row-center_y)^2/radius_y^2 + (col-center_x)^2/radius_x^2) > 1
                cropped_im(row,col) = nan;
            end
        end
    end
    
    % Necessary because of potential DIV/0 behavior
    if radius_x == 0 || radius_y == 0
        cropped_im = nan(size(cropped_im));
    end

end