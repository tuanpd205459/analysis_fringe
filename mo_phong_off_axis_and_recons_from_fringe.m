%% Off-Axis Holography Simulation & 3D Surface Reconstruction from Fringes
% -------------------------------------------------------------------------
%
% This script demonstrates a complete workflow for simulating off-axis 
% holography, extracting interference fringes, and reconstructing the 3D 
% surface from those fringes. The process is divided into two main parts:
%
% PART 1: Off-axis holography simulation
% - Simulate object and reference waves
% - Generate and visualize the interference pattern (hologram)
% - Convert phase to surface height
%
% PART 2: Fringe extraction and 3D surface reconstruction
% - Extract/fringe binarization
% - Skeletonize (thinning) for single-pixel wide fringes
% - Estimate fringe orientation, rotate and label fringes
% - Reconstruct the 3D surface and level it
%
% NOTE: For educational and research purposes.
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
ampPhase = 4;      % Amplitude of phase modulation
noise = 0;         % Add noise if needed (currently not used)
[x, y] = meshgrid(linspace(-1,1,N));
phi_obj = ampPhase * exp(-10*(x.^2 + y.^2)); % Gaussian phase profile

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

% --- 5. Reconstruct 3D surface from fringes ---
lambda = 632.8e-9; % Wavelength used above
khoang_cach_van = (lambda / 2)/cosd(abs(avg_angle)); % Fringe-to-height mapping

BW = BW_rotated;
L = bwlabel(BW);
num_labels = max(L(:));
X = []; Y = []; Z = [];

for i = 1:num_labels
    [y, x] = find(L == i); 
    z = ones(size(x)) * (i-1) * khoang_cach_van;
    X = [X; x];
    Y = [Y; y];
    Z = [Z; z];
end

%% Interpolate to get a smooth 3D surface
[xq, yq] = meshgrid(1:size(BW,2), 1:size(BW,1));
F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
Zq = F(xq, yq);
Zq(~isfinite(Zq)) = 0;

figure('Name', 'Reconstructed 3D Surface');
surf(xq, yq, Zq, 'EdgeColor', 'none');
colormap turbo;
colorbar;
xlabel('X (px)'); ylabel('Y (px)'); zlabel('Height (m)');
title('3D Surface Reconstructed from Fringes');
view([45 30]);
c = colorbar; 
c.Label.String = 'Height (m)';

%% --- 6. Level the reconstructed surface (remove tilt) ---
% Crop for better display
Z = Zq(100:end-100, 100:end-100);
[M, N] = size(Z);
[xGrid, yGrid] = meshgrid(1:N, 1:M);
x = xGrid(:);
y = yGrid(:);
z = Z(:);

%% Fit and remove tilt (plane subtraction)
A = [x, y, ones(size(x))];  
coeff = A \ z;              
Z_fit = reshape(A * coeff, size(Z));
Z_leveled = Z - Z_fit;  

%% Normalize Z to start from zero and invert if necessary
Z_inverted = -Z_leveled;
Z_offset = Z_inverted - min(Z_inverted(:));

figure('Name', 'Tilt-Removed (Leveled) 3D Surface');
surf(xGrid, yGrid, Z_offset);
shading interp;
title('3D Surface after Tilt Removal (Leveled)');
xlabel('X (px)');
ylabel('Y (px)');
zlabel('Height (m)');
colormap parula;
colorbar;