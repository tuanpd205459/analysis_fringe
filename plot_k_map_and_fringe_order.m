clc, clear, close all;
addpath("D:\tuan\analysis\analysis_fringe\export_fig");

load("cchuan_phase_comparison.mat");

%% 1. INPUT DATA & COORDINATES

% 1.5 SETUP SPATIAL COORDINATES (MM)
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(kMap);
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

% 2. ROBUST LIMITS CALCULATION
% Tính toán giới hạn màu dựa trên phân vị để loại bỏ điểm nhiễu (hot pixels)
robust_min = prctile(kMap(:), 0.1); 
robust_max = prctile(kMap(:), 99.8); 
z_lims = [robust_min, robust_max];

%% 3. FIGURE SETTINGS
figWidth  = 17.5;  % Chiều rộng hình (cm)
figHeight = 10;    % Chiều cao hình (cm)
fontSize  = 10;
fontName  = 'Times New Roman';

% Tạo Figure
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_kMap_Visualization', ...
             'NumberTitle', 'off');

% Sử dụng Tiledlayout để quản lý khoảng cách tốt hơn (thay thế subplot cũ)
t = tiledlayout(1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

%% 4. PLOTTING
nexttile;

% Vẽ ảnh
imagesc(x_vec, y_vec, kMap);
axis image;
clim(z_lims);       % Áp dụng giới hạn màu đã tính
colormap(gca, turbo);

% Thiết lập trục và nhãn
% title('(a) Reconstructed Phase', 'FontWeight','bold', 'FontSize', fontSize+1, ...
%       'FontName', fontName, 'Interpreter','latex');

xlabel('$x$ (mm)', 'Interpreter', 'latex');
ylabel('$y$ (mm)', 'Interpreter', 'latex');

set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'LineWidth', 1, 'TickLabelInterpreter', 'latex');
box on;

% 5. COLORBAR SETTINGS
cb = colorbar;
cb.Layout.Tile = 'east'; 
cb.Limits = z_lims;
cb.TickLabelInterpreter = 'latex';
cb.FontSize = fontSize;

% Nhãn Colorbar
cb.Label.String = 'K-order'; 
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize + 1;

%% 6. EXPORT
saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% Tạo timestamp để tên file không bị trùng
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = ['Fig_kMap_2D_turbo_' timestamp]; 
fullPath = fullfile(saveFolder, fileName);

% Xuất hình (Yêu cầu cài đặt export_fig hoặc dùng exportgraphics của Matlab)
try
    % Cách 1: Sử dụng export_fig (nếu đã cài đặt toolbox này)
    export_fig([fullPath '.png'], '-png', '-r600'); 
    % export_fig([fullPath '.eps'], '-eps', '-opengl'); 
    fprintf('Exported using export_fig: %s\n', fullPath);
catch
    % Cách 2: Sử dụng exportgraphics (Native Matlab - Khuyến nghị cho Matlab 2020a+)
    exportgraphics(fig, [fullPath '.png'], 'Resolution', 600);
    % exportgraphics(fig, [fullPath '.eps'], 'ContentType', 'vector');
    fprintf('Exported using exportgraphics: %s\n', fullPath);
end


%% ve anh 3D
%% 1. INPUT DATA & CALCULATIONS
% Giả sử kMap đã có trong workspace
% kMap = ...; 

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
yline(y_loc_mm, '--w', 'LineWidth', 1.5, 'Alpha', 0.8); 
hold off;

title('(a) 2D Phase Map', 'Interpreter','latex');
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

title('(b) 3D Visualization', 'Interpreter','latex');
xlabel('$x$ (mm)', 'Interpreter', 'latex');
ylabel('$y$ (mm)', 'Interpreter', 'latex');
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
fileName = ['Fig_Analysis_Full_' timestamp];
fullPath = fullfile(saveFolder, fileName);

try
    exportgraphics(fig, [fullPath '.png'], 'Resolution', 600);
    fprintf('Exported: %s.png\n', fileName);
catch
    saveas(gcf, [fullPath '.png']);
end




%%  Hieenr thi anha skeleton
% 1. INPUT DATA & COORDINATES

% 1.5 SETUP SPATIAL COORDINATES (MM)
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(BW);
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

% 2. ROBUST LIMITS CALCULATION
% Tính toán giới hạn màu dựa trên phân vị để loại bỏ điểm nhiễu (hot pixels)
robust_min = prctile(BW(:), 0.1); 
robust_max = prctile(BW(:), 99.8); 
z_lims = [robust_min, robust_max];

%% 3. FIGURE SETTINGS
figWidth  = 17.5;  % Chiều rộng hình (cm)
figHeight = 10;    % Chiều cao hình (cm)
fontSize  = 10;
fontName  = 'Times New Roman';

% Tạo Figure
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_skeleton_Visualization', ...
             'NumberTitle', 'off');

% Sử dụng Tiledlayout để quản lý khoảng cách tốt hơn (thay thế subplot cũ)
t = tiledlayout(1, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

%% 4. PLOTTING
nexttile;

% Vẽ ảnh
imagesc(x_vec, y_vec, kMap);
axis image;
clim(z_lims);       % Áp dụng giới hạn màu đã tính
colormap(gca, turbo);

% Thiết lập trục và nhãn
% title('(a) Reconstructed Phase', 'FontWeight','bold', 'FontSize', fontSize+1, ...
%       'FontName', fontName, 'Interpreter','latex');

xlabel('$x$ (mm)', 'Interpreter', 'latex');
ylabel('$y$ (mm)', 'Interpreter', 'latex');

set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
    'LineWidth', 1, 'TickLabelInterpreter', 'latex');
box on;

% % 5. COLORBAR SETTINGS
% cb = colorbar;
% cb.Layout.Tile = 'east'; 
% cb.Limits = z_lims;
% cb.TickLabelInterpreter = 'latex';
% cb.FontSize = fontSize;
% 
% % Nhãn Colorbar
% cb.Label.String = 'Skeleton'; 
% cb.Label.Interpreter = 'latex';
% cb.Label.FontSize = fontSize + 1;

%% 6. EXPORT
saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% Tạo timestamp để tên file không bị trùng
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = ['Fig_skeleton_2D_turbo_' timestamp]; 
fullPath = fullfile(saveFolder, fileName);

% Xuất hình (Yêu cầu cài đặt export_fig hoặc dùng exportgraphics của Matlab)
try
    % Cách 1: Sử dụng export_fig (nếu đã cài đặt toolbox này)
    export_fig([fullPath '.png'], '-png', '-r600'); 
    % export_fig([fullPath '.eps'], '-eps', '-opengl'); 
    fprintf('Exported using export_fig: %s\n', fullPath);
catch
    % Cách 2: Sử dụng exportgraphics (Native Matlab - Khuyến nghị cho Matlab 2020a+)
    exportgraphics(fig, [fullPath '.png'], 'Resolution', 600);
    % exportgraphics(fig, [fullPath '.eps'], 'ContentType', 'vector');
    fprintf('Exported using exportgraphics: %s\n', fullPath);
end