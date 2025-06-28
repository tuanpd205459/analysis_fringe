%% Off-Axis Holography Simulation & 3D Surface Reconstruction from Fringes
% -------------------------------------------------------------------------
% Thay đổi fitting bằng Zernike
% -------------------------------------------------------------------------

clc; clear; close all;

%% ==== PART 1: Off-axis Holography Simulation ====
% Image size (number of CCD pixels)
Ax = 1080;
Ay = 1080;
N = 1080;

% Coordinate grid for simulation
[Xa, Ya] = meshgrid(1:Ax, 1:Ay);

% Create a parabolic object phase (as a Gaussian for generality)
ampPhase = 10;      % Amplitude of phase modulation
[x, y] = meshgrid(linspace(-1,1,N));
noise_level = 0; % Độ lớn của nhiễu
phi_obj = ampPhase * exp(-10*(x.^2 + y.^2)) + noise_level*randn(size(x));

% Generate object wave
Es = exp(1i * phi_obj);

% Show object phase and the corresponding phase surface
figure('Name', 'Object Wave Visualization');
subplot(1,2,1)
imagesc(angle(Es));
title('Object Wave Phase (\phi_{obj})');
axis square; colormap(hsv); colorbar; axis off;

subplot(1,2,2)
surf(phi_obj, 'EdgeColor', 'none');
title('Object Phase Surface (\phi_{obj})');
xlabel('x (px)'); ylabel('y (px)'); zlabel('\phi');
colormap(jet); colorbar; view([45 30]);

% Reference wave parameters (off-axis)
lambda = 632.8e-9; % Wavelength in meters (HeNe red)
theta = 5 * pi / 180;   % Off-axis angle in radians
k = 2 * pi / lambda;
kSinTheta = k * sin(theta);
scale_ref = 1e-7; % Scaling for reference phase spatial frequency
phi_ref = scale_ref * kSinTheta * Xa;
E0 = exp(1i * phi_ref);  % Reference wave, tilted along x-axis

% Convert phase to surface height (in meters)
h_surface = (lambda/(4*pi)) * phi_obj; % Surface height (m)
%%
% Visualize reference phase and phase surface
figure('Name', 'Reference Wave Visualization');
subplot(1,2,1)
imagesc(angle(E0));
title('Reference Wave Phase (\phi_{ref})');
axis square; colormap(jet); colorbar; axis off;

subplot(1,2,2)
surf(phi_ref, 'EdgeColor', 'none');
title('Reference Phase Surface (\phi_{ref})');
xlabel('x (px)'); ylabel('y (px)'); zlabel('\phi_{ref}');
colormap(jet); colorbar; view([45 30]);

% Simulate interference pattern (hologram)
I = abs(E0 + Es).^2;
%%
% --- Visualization of simulation results ---
figure('Name', 'Holography Simulation Results', 'Position', [200, 300, 1200, 400]);
subplot(1,3,1)
imagesc(angle(Es)); axis square; colormap(hsv); colorbar; axis off;
title('Object Wave Phase (\phi_{obj})');

subplot(1,3,2)
imagesc(angle(E0)); axis square; colormap(jet); colorbar; axis off;
title('Reference Wave Phase (\phi_{ref})');

subplot(1,3,3)
imagesc(I); axis square; colormap(gray); axis off;
title('Off-axis Interference Pattern (Hologram)');

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

% Optional: 2D colormap height
% figure; imagesc(h_surface); axis square; colormap(jet); colorbar; axis off;
% title('Object Surface Height (True, 2D View)');

%% ==== PART 2: Fringe Extraction & 3D Surface Reconstruction ====

% --- 1. Convert to grayscale and binarize ---
Img_Original = I;
if size(Img_Original, 3) == 3
    grayImg = rgb2gray(Img_Original);
else
    grayImg = Img_Original;
end
img = im2double(grayImg);

% Otsu thresholding (invert so fringe = 1, background = 0)
thresh = graythresh(grayImg);
BW_Original = ~imbinarize(grayImg, thresh);

% --- 2. Skeletonization (Thinning) ---
changing = 1;
[rows, columns] = size(BW_Original);
BW_Thinned = BW_Original;

while changing
    BW_Del = ones(rows, columns);
    changing = 0;
    % Step 1
    for i=2:rows-1
        for j = 2:columns-1
            P = [BW_Thinned(i,j) BW_Thinned(i-1,j) BW_Thinned(i-1,j+1) BW_Thinned(i,j+1) BW_Thinned(i+1,j+1) ...
                BW_Thinned(i+1,j) BW_Thinned(i+1,j-1) BW_Thinned(i,j-1) BW_Thinned(i-1,j-1) BW_Thinned(i-1,j)];
            if (BW_Thinned(i,j) == 1 && sum(P(2:end-1))<=6 && sum(P(2:end-1)) >=2 && ...
                    P(2)*P(4)*P(6)==0 && P(4)*P(6)*P(8)==0)
                A = 0;
                for k = 2:9
                    if P(k) == 0 && P(k+1)==1
                        A = A+1;
                    end
                end
                if (A==1)
                    BW_Del(i,j)=0;
                    changing = 1;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned.*BW_Del;

    % Step 2
    BW_Del = ones(rows, columns);
    for i=2:rows-1
        for j = 2:columns-1
            P = [BW_Thinned(i,j) BW_Thinned(i-1,j) BW_Thinned(i-1,j+1) BW_Thinned(i,j+1) BW_Thinned(i+1,j+1) ...
                BW_Thinned(i+1,j) BW_Thinned(i+1,j-1) BW_Thinned(i,j-1) BW_Thinned(i-1,j-1) BW_Thinned(i-1,j)];
            if (BW_Thinned(i,j) == 1 && sum(P(2:end-1))<=6 && sum(P(2:end-1)) >=2 && ...
                    P(2)*P(4)*P(8)==0 && P(2)*P(6)*P(8)==0)
                A = 0;
                for k = 2:9
                    if P(k) == 0 && P(k+1)==1
                        A = A+1;
                    end
                end
                if (A==1)
                    BW_Del(i,j)=0;
                    changing = 1;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned.*BW_Del;
end

BW = BW_Thinned;
vung_chon = BW;

% --- 3. Estimate fringe orientation using Hough Transform ---
[H, theta, rho] = hough(vung_chon);
P = houghpeaks(H, 5);
lines = houghlines(vung_chon, theta, rho, P);

%% Compute average fringe angle
avg_angle = mean([lines.theta]);
goc_vuong_goc = avg_angle + 90; % Perpendicular direction to fringes

[H, W] = size(vung_chon);
x_center = W / 2;
y_center = H / 2;
slope = tand(goc_vuong_goc);
x1 = 1; x2 = W;
y1 = y_center + slope * (x1 - x_center);
y2 = y_center + slope * (x2 - x_center);

%% --- 4. Visualize skeletonized and rotated fringes ---
% Rotate binary fringe image so fringes are nearly vertical
angle_deg = -90 + goc_vuong_goc;
BW_rotated = imrotate(BW, angle_deg, 'bilinear', 'crop');
BW_rotated = BW_rotated(50:end-50, 50:end-50); % Crop border

figure('Name', 'Skeletonized and Rotated Fringes');
imshow(BW_rotated);
title('Skeletonized Fringes (Rotated, Cropped)');
hold on;

% Find boundaries and label each fringe
[B,L] = bwboundaries(BW_rotated, 'noholes');
for k = 1:length(B)
    boundary = B{k};
    c = mean(boundary(:,2));  % centroid x
    r = mean(boundary(:,1));  % centroid y
    text(c, r, num2str(k), 'Color','yellow','FontSize',12,'FontWeight','bold');
end
hold off;

%% --- 5. Reconstruct 3D surface from fringes ---
% lambda = 632.8e-9; % Wavelength used above
% khoang_cach_van = (lambda / 2)/cosd(abs(avg_angle)); % Fringe-to-height mapping
%
% BW = BW_rotated;
% L = bwlabel(BW);
% num_labels = max(L(:));
% X = []; Y = []; Z = [];
%
% for i = 1:num_labels
%     [y, x] = find(L == i);
%     z = ones(size(x)) * (i-1) * khoang_cach_van;
%     X = [X; x];
%     Y = [Y; y];
%     Z = [Z; z];
% end

% 5 Interpolate to get a smooth 3D surface
% [xq, yq] = meshgrid(1:size(BW,2), 1:size(BW,1));
% F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
% Zq = F(xq, yq);
% Zq(~isfinite(Zq)) = 0;
%
% figure('Name', 'Reconstructed 3D Surface');
% surf(xq, yq, Zq, 'EdgeColor', 'none');
% colormap turbo;
% colorbar;
% xlabel('X (px)'); ylabel('Y (px)'); zlabel('Height (m)');
% title('3D Surface Reconstructed from Fringes');
% view([45 30]);
% c = colorbar;
% c.Label.String = 'Height (m)';
%% --- 5. Dựng lại bề mặt 3D từ nhiễu vân bằng fitting đa thức Zernike ---
lambda = 632.8e-9;  % Bước sóng (m)
khoang_cach_van = (lambda / 2) / cosd(abs(avg_angle));  % Mapping từ fringe → chiều cao

% Tìm nhãn các vùng nhiễu vân
BW = BW_rotated;
L = bwlabel(BW);
num_labels = max(L(:));

X = []; Y = []; Z = [];

% Duyệt qua từng nhãn để tạo điểm 3D
for i = 1:num_labels
    [y, x] = find(L == i);
    z = ones(size(x)) * (i-1) * khoang_cach_van;
    X = [X; x(:)];
    Y = [Y; y(:)];
    Z = [Z; z(:)];
end

%% Chuẩn hóa điểm vào đĩa đơn vị [-1, 1]
Xn = 2 * (X - min(X)) / (max(X) - min(X)) - 1;
Yn = 2 * (Y - min(Y)) / (max(Y) - min(Y)) - 1;
R = sqrt(Xn.^2 + Yn.^2);
inside = R <= 1;

% Giữ lại các điểm nằm trong đĩa đơn vị
Xn = Xn(inside);
Yn = Yn(inside);
Zn = Z(inside);

%% Tạo ma trận cơ sở Zernike
maxN = 10;  % Bậc Zernike tối đa
B = [];
[nv, mv] = zernikeOrders(maxN);  % Hàm này bạn phải có sẵn

for i = 1:length(nv)
    Zm = zernfun(nv(i), mv(i), Xn, Yn);  % zernfun là hàm dựng hàm cơ sở Zernike
    B = [B, Zm(:)];
end

%% Fit hệ số Zernike bằng least squares
c = B \ Zn;

%% Dựng lại bề mặt từ hệ số Zernike
[xq, yq] = meshgrid(linspace(-1, 1, size(BW,2)), linspace(-1, 1, size(BW,1)));
rq = sqrt(xq.^2 + yq.^2);
mask = rq <= 1;

Zq = zeros(size(xq));
for i = 1:length(c)
    Zq = Zq + c(i) * zernfun(nv(i), mv(i), xq, yq);
end
Zq(~mask) = NaN;  % Loại bỏ điểm ngoài đĩa đơn vị

% Hiển thị bề mặt dựng lại
figure('Name', '3D Surface Reconstructed by Zernike Fit');
surf(xq, yq, Zq, 'EdgeColor', 'none');
colormap turbo; colorbar;
xlabel('X (norm)'); ylabel('Y (norm)'); zlabel('Height (m)');
title('3D Surface Reconstructed from Fringes (Zernike fit)');
view([45 30]);
c = colorbar;
c.Label.String = 'Height (m)';


%% --- 6. Level the reconstructed surface (remove tilt) ---
% Crop for better display (nên kiểm tra kích thước trước khi crop)
crop_val = 100;
if size(Zq,1) > 2*crop_val && size(Zq,2) > 2*crop_val
    Z = Zq(crop_val+1:end-crop_val, crop_val+1:end-crop_val);
else
    Z = Zq;
end
[M, N] = size(Z);
[xGrid, yGrid] = meshgrid(1:N, 1:M);
z = Z(:);
good = ~isnan(z); % Chỉ lấy điểm hợp lệ

%% Fit and remove tilt (plane subtraction)
A = [xGrid(:), yGrid(:), ones(numel(z),1)];
coeff = A(good,:) \ z(good);
Z_fit = reshape(A * coeff, size(Z));
Z_leveled = Z - Z_fit;

%% Normalize Z to start from zero and invert if necessary
Z_inverted = -Z_leveled;
Z_offset = Z_inverted - min(Z_inverted(:));

figure('Name', 'Tilt-Removed (Leveled) 3D Surface');
surf(xGrid, yGrid, Z_offset, 'EdgeColor', 'none');
shading interp;
title('3D Surface after Tilt Removal (Leveled)');
xlabel('X (px)');
ylabel('Y (px)');
zlabel('Height (m)');
colormap parula;
colorbar;

%% ước lượng k_est
unwrapped_phase_est = Z_offset*4*pi/lambda;
wrapped_phase_est = atan2(sin(unwrapped_phase_est),cos(unwrapped_phase_est));
k_est = round((unwrapped_phase_est - wrapped_phase_est) / (2*pi));
k_est(isnan(k_est)) = 0;

figure;
surf(wrapped_phase_est, 'EdgeColor', 'none');
title('Estimated wrapped Phase Object (\phi_{wrapped-est})');
xlabel('x (px)'); ylabel('y (px)'); zlabel('\phi_{wrapped-est}');
colormap turbo; colorbar;

figure;
plot(wrapped_phase_est(round(end/2), :)); % Plots the middle row as a line
title("Mặt cắt ngang wrapped phase");
xlabel('Pixel');
ylabel('Phase');

figure;
surf(k_est, 'EdgeColor', 'none'); % Plots the middle row as a line
title("k est");
xlabel('Pixel');
ylabel('k');colormap turbo; colorbar;

figure;
surf(phi_obj, 'EdgeColor', 'none');
title('True Phase Object (\phi_{true})');
xlabel('x (px)'); ylabel('y (px)'); zlabel('\phi_{true}');
colormap turbo; colorbar;

figure;
surf(unwrapped_phase_est, 'EdgeColor', 'none');
title('Estimated UW Phase Object (\phi_{true})');
xlabel('x (px)'); ylabel('y (px)'); zlabel('\phi_{est}');
colormap turbo; colorbar;


%% Tham chiếu cho unwrapping pha truyền thống

% --- Bước 7: Truyền initial guess vào unwrap truyền thống --- Nếu bạn có
% hàm unwrap2D LS hoặc Quality-Guided

phi_unwrapped = unwrap_LS_FD_DCT(phi_wrapped);

% --- Optional: Điều chỉnh theo k_est nếu cần --- 
phi_unwrapped_refined = phi_unwrapped + 2*pi* k_est;

% Trả về kết quả 
phi_unwrapped = phi_unwrapped_refined;










%% Hàm phụ trợ
function Z = zernfun(n, m, x, y)
% Zernike polynomial of order (n,m) at points (x,y) on unit disk
% x, y: same size vectors or matrices
% n, m: integers, n >= 0, |m| <= n, n-m chẵn
[theta, r] = cart2pol(x, y);
if m >= 0
    Z = zernikeRadial(n, m, r) .* cos(m*theta);
else
    Z = zernikeRadial(n, -m, r) .* sin(-m*theta);
end
Z(r>1) = 0;
end

function R = zernikeRadial(n, m, r)
% Radial polynomial R_n^m(r)
R = zeros(size(r));
for s = 0:((n-m)/2)
    c = (-1)^s * factorial(n-s) / (factorial(s)*factorial((n+m)/2-s)*factorial((n-m)/2-s));
    R = R + c * r.^(n-2*s);
end
end
%%
function [nList, mList] = zernikeOrders(maxN)
% Output vector nList, mList for all Zernike terms up to order maxN
nList = [];
mList = [];
for n = 0:maxN
    for m = -n:2:n
        nList(end+1) = n;
        mList(end+1) = m;
    end
end
end