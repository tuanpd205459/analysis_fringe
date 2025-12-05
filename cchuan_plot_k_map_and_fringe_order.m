clc, clear, close all;
addpath("D:\tuan\analysis\analysis_fringe\export_fig");

load("cchuan_phase_comparison.mat");

BW = BW_connected;
hologram_original = rot90(hologram_original, 1);
%% ve anh 3D - mat MCN
% 1. INPUT DATA & CALCULATIONS
% 1.1 Setup Coordinates
px_size = 3.45e-3; % mm
[rows, cols] = size(kMap);
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

% 1.2 Calculate Limits (Robust)
robust_min = prctile(kMap(:), 0.1); 
robust_max = prctile(kMap(:), 99.8); 
z_lims = [robust_min, robust_max];

% 1.3 Extract Profile Data (Cắt ngang tại dòng giữa)
mid_row_idx = round(rows / 2);      % Chỉ số dòng giữa
y_loc_mm = y_vec(mid_row_idx);      % Tọa độ y thực tế (mm)
profile_data = kMap(mid_row_idx, :); % Dữ liệu profile

%% 2. FIGURE SETTINGS
figWidth  = 18;    % Rộng hơn chút để chứa đủ 3 biểu đồ
figHeight = 14;    % Cao hơn để chứa 2 hàng
fontSize  = 10;
fontName  = 'Times New Roman';

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Analysis_2D_3D_Profile', ...
             'NumberTitle', 'off');

% TiledLayout: 2 hàng, 2 cột
t = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

%% 3. PLOT 1: 2D IMAGE (Top Left)
ax1 = nexttile;
imagesc(x_vec, y_vec, kMap);
axis image;
clim(z_lims);
colormap(ax1, turbo);

% Vẽ đường đứt nét màu trắng/đen để chỉ vị trí cắt profile
hold on;
yline(y_loc_mm, '--w', 'LineWidth', 2); 
hold off;

title('(a) 2D k-order', 'Interpreter','latex');
ylabel('$y$ (mm)', 'Interpreter', 'latex');
xlabel('$x$ (mm)', 'Interpreter', 'latex'); % Thêm nhãn x cho rõ ràng
set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex', 'YDir', 'normal'); % YDir normal để 0 ở dưới (nếu muốn)

%% 4. PLOT 2: 3D SURFACE (Top Right)
ax2 = nexttile;
s = surf(x_vec, y_vec, kMap);
s.EdgeColor = 'none'; % Tắt lưới đen trên mặt 3D để mượt hơn
s.FaceColor = 'interp';

% Setup view và ánh sáng
view(-45, 30); % Góc nhìn nghiêng tiêu chuẩn
camlight; lighting gouraud; % Tạo khối 3D đẹp hơn
axis tight; 
clim(z_lims);
colormap(ax2, turbo);

title('(b) 3D k-order', 'Interpreter','latex');
hx = xlabel('$x$ (mm)', 'Interpreter', 'latex');
set(hx, 'Rotation', 12.5);                % Xoay nghiêng (chỉnh số này nếu cần)
set(hx, 'VerticalAlignment', 'middle');
set(hx, 'HorizontalAlignment', 'left'); % Căn lề trái để bám theo trục

% Trục Y
hy = ylabel('$y$ (mm)', 'Interpreter', 'latex');
set(hy, 'Rotation', -10);               % Xoay nghiêng ngược lại
set(hy, 'VerticalAlignment', 'middle');
set(hy, 'HorizontalAlignment', 'right'); % Căn lề phải để bám theo trục

zlabel('k-order', 'Interpreter', 'latex');
set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex');

%% 5. COLORBAR (Chung cho hàng trên)
% Thêm colorbar gắn với trục 3D hoặc dùng chung
cb = colorbar;
cb.Layout.Tile = 'east'; % Đặt bên phải ngoài cùng
cb.Limits = z_lims;
cb.TickLabelInterpreter = 'latex';
cb.Label.String = 'k-order';
cb.Label.Interpreter = 'latex';

%% 6. PLOT 3: CROSS-SECTION PROFILE (Bottom - Spanning 2 columns)
ax3 = nexttile([1, 2]); % Chiếm 1 hàng, 2 cột
plot(x_vec, profile_data, 'b-', 'LineWidth', 1.5);
grid on;
box on;

% Giới hạn trục Y theo z_lims để đồng bộ, hoặc để auto
ylim(z_lims); 
xlim([min(x_vec), max(x_vec)]);

% Tiêu đề có chứa thông tin vị trí cắt
title(['(c) Cross-section Profile at $y = ' sprintf('%.2f', y_loc_mm) '$ mm'], ...
      'Interpreter','latex');
xlabel('$x$ (mm)', 'Interpreter', 'latex');
ylabel('k-order', 'Interpreter', 'latex');

set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex');

%% 7. EXPORT
saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir'); mkdir(saveFolder); end

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = ['Fig_Analysis_Full_kMap' timestamp];
fullPath = fullfile(saveFolder, fileName);

try
    exportgraphics(fig, [fullPath '.png'], 'Resolution', 600);
    fprintf('Exported: %s.png\n', fileName);
catch
    saveas(gcf, [fullPath '.png']);
end

%% dung don vi la pixel - anh k-map

[rows, cols] = size(kMap);
x_vec = 0:cols-1;
y_vec = 0:rows-1;

% 1.2 Calculate Limits (Robust)
robust_min = prctile(kMap(:), 0.1); 
robust_max = prctile(kMap(:), 99.8); 
z_lims = [robust_min, robust_max];

% 1.3 Extract Profile Data (Cắt ngang tại dòng giữa)
mid_row_idx = round(rows / 2);      % Chỉ số dòng giữa
y_loc_mm = y_vec(mid_row_idx);      % Tọa độ y thực tế (mm)
profile_data = kMap(mid_row_idx, :); % Dữ liệu profile

%% 2. FIGURE SETTINGS
figWidth  = 18;    % Rộng hơn chút để chứa đủ 3 biểu đồ
figHeight = 14;    % Cao hơn để chứa 2 hàng
fontSize  = 10;
fontName  = 'Times New Roman';

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Analysis_2D_3D_Profile', ...
             'NumberTitle', 'off');

% TiledLayout: 2 hàng, 2 cột
t = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

%% 3. PLOT 1: 2D IMAGE (Top Left)
ax1 = nexttile;
imagesc(x_vec, y_vec, kMap);
axis image;
clim(z_lims);
colormap(ax1, turbo);

% Vẽ đường đứt nét màu trắng/đen để chỉ vị trí cắt profile
hold on;
yline(y_loc_mm, '--w', 'LineWidth', 2); 
hold off;

title('(a) 2D k-order', 'Interpreter','latex');
ylabel('$y$ (pixel)', 'Interpreter', 'latex');
xlabel('$x$ (pixel)', 'Interpreter', 'latex'); % Thêm nhãn x cho rõ ràng
set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex', 'YDir', 'normal'); % YDir normal để 0 ở dưới (nếu muốn)

%% 4. PLOT 2: 3D SURFACE (Top Right)
ax2 = nexttile;
s = surf(x_vec, y_vec, kMap);
s.EdgeColor = 'none'; % Tắt lưới đen trên mặt 3D để mượt hơn
s.FaceColor = 'interp';

% Setup view và ánh sáng
view(-45, 30); % Góc nhìn nghiêng tiêu chuẩn
camlight; lighting gouraud; % Tạo khối 3D đẹp hơn
axis tight; 
clim(z_lims);
colormap(ax2, turbo);

title('(b) 3D k-order', 'Interpreter','latex');
% xlabel('$x$ (pixel)', 'Interpreter', 'latex');
% ylabel('$y$ (pixel)', 'Interpreter', 'latex');

hx = xlabel('$x$ (pixel)', 'Interpreter', 'latex');
set(hx, 'Rotation', 12.5);                % Xoay nghiêng (chỉnh số này nếu cần)
set(hx, 'VerticalAlignment', 'middle');
set(hx, 'HorizontalAlignment', 'left'); % Căn lề trái để bám theo trục

% Trục Y
hy = ylabel('$y$ (pixel)', 'Interpreter', 'latex');
set(hy, 'Rotation', -10);               % Xoay nghiêng ngược lại
set(hy, 'VerticalAlignment', 'middle');
set(hy, 'HorizontalAlignment', 'right'); % Căn lề phải để bám theo trục


zlabel('k-order', 'Interpreter', 'latex');
set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex');

%% 5. COLORBAR (Chung cho hàng trên)
% Thêm colorbar gắn với trục 3D hoặc dùng chung
cb = colorbar;
cb.Layout.Tile = 'east'; % Đặt bên phải ngoài cùng
cb.Limits = z_lims;
cb.TickLabelInterpreter = 'latex';
cb.Label.String = 'k-order';
cb.Label.Interpreter = 'latex';

%% 6. PLOT 3: CROSS-SECTION PROFILE (Bottom - Spanning 2 columns)
ax3 = nexttile([1, 2]); % Chiếm 1 hàng, 2 cột
plot(x_vec, profile_data, 'b-', 'LineWidth', 1.5);
grid on;
box on;

% Giới hạn trục Y theo z_lims để đồng bộ, hoặc để auto
ylim(z_lims); 
xlim([min(x_vec), max(x_vec)]);

% Tiêu đề có chứa thông tin vị trí cắt
title(['(c) Cross-section Profile at $y = ' sprintf('%.2f', y_loc_mm) '$ pixel'], ...
      'Interpreter','latex');
xlabel('$x$ (pixel)', 'Interpreter', 'latex');
ylabel('k-order', 'Interpreter', 'latex');

set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex');

%% 7. EXPORT
saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir'); mkdir(saveFolder); end

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = ['Fig_Analysis_Full_kMap_mm' timestamp];
fullPath = fullfile(saveFolder, fileName);

try
    exportgraphics(fig, [fullPath '.png'], 'Resolution', 600);
    fprintf('Exported: %s.png\n', fileName);
catch
    saveas(gcf, [fullPath '.png']);
end




%%  Hien thi anh skeleton - cos mau

% 1.1 Coordinates
px_size = 3.45e-3; 
[rows, cols] = size(BW);
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

% 1.2 Processing for Right Image (Skeleton Visualization)
% Làm dày nét một chút để dễ nhìn thấy trên hình vẽ (nếu muốn nét mảnh gốc thì bỏ dòng này)
se = strel('square', 3); 
BW_vis = imdilate(BW, se); 

%% 2. FIGURE SETTINGS
figWidth  = 18;    
figHeight = 9;     
fontSize  = 10;
fontName  = 'Times New Roman';

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_SideBySide_NoOverlay', ...
             'NumberTitle', 'off');

% Layout 1 hàng, 2 cột
t = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

%% 3. PLOT LEFT: ORIGINAL FRINGE
ax1 = nexttile;
imagesc(x_vec, y_vec, hologram_original);
axis image;
colormap(ax1, "gray"); 

title('(a) Hologram image', 'Interpreter', 'latex');
xlabel('$x$ (mm)', 'Interpreter', 'latex');
ylabel('$y$ (mm)', 'Interpreter', 'latex');
set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex');

%% 4. PLOT RIGHT: SKELETON ONLY (BINARY)
ax2 = nexttile;
imagesc(x_vec, y_vec, BW_vis); 
axis image;

% Thiết lập màu: Nền trắng, Xương đen
colormap(ax2, flipud(gray)); 
clim(ax2, [0 1]); % Khóa giới hạn màu nhị phân

title('(b) Extracted Skeleton', 'Interpreter', 'latex');
xlabel('$x$ (mm)', 'Interpreter', 'latex');
ylabel('$y$ (mm)', 'Interpreter', 'latex');
set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'TickLabelInterpreter', 'latex');
box on; 

%% 5. SYNC ZOOM (Đồng bộ)
linkaxes([ax1, ax2], 'xy'); 
% Khi zoom hình trái, hình phải sẽ zoom theo đúng vị trí đó.

%% 6. EXPORT
saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir'); mkdir(saveFolder); end

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = ['Fig_SideBySide_Binary_' timestamp]; 
fullPath = fullfile(saveFolder, fileName);

try
    exportgraphics(fig, [fullPath '.png'], 'Resolution', 600);
    fprintf('Exported: %s.png\n', fullPath);
catch
    export_fig([fullPath '.png'], '-png', '-r600'); 
end

