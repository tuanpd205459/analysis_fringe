% phase comparison simulation
clc, clear, close all;
addpath("D:\tuan\analysis\analysis_fringe\export_fig");
%%
load("draft_ss_anh_mo_phong.mat");

%%
% [rows, cols] = size(object_phase_without_noise);
% cx = cols/2;
% cy = rows/2;
% [x, y] = meshgrid(1:cols, 1:rows);
% x = x - cx;
% y = y - cy;
% 
% [theta, rho] = cart2pol(x, y);
% 
% R = min(cx, cy);   
% mask = rho <= R;
% 
% object_phase_without_noise(~mask) = NaN;
object_phase_without_noise = object_phase_without_noise - min(object_phase_without_noise(:));
final_phi_goldstein = final_phi_goldstein - min(final_phi_goldstein(:));
final_phi_quality = final_phi_quality - min(final_phi_quality(:));
final_phi_tie_dct = final_phi_tie_dct - min(final_phi_tie_dct(:));
final_phi_wls = final_phi_wls - min(final_phi_wls(:));
final_phi_proposed = final_phi_proposed - min(final_phi_proposed(:));

%% 1. PACK RESULTS (ĐÃ BỎ 'PROPOSED RAW')
dataList = { ...
    object_phase_without_noise, 'anh ban dau';...
    final_phi_goldstein,         'Goldstein'; ...
    final_phi_quality,           'Quality-Guided'; ...
    final_phi_tie_dct,           'TIE-DCT'; ...
    final_phi_wls,               'WLS'; ...
    final_phi_proposed,          'Proposed (Final)' ...
};

% Ảnh màu turbo
%% 1.5 SETUP SPATIAL COORDINATES (MM)
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList{1,1});
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

%% 2. GLOBAL COLOR LIMITS

all_pixels = []; 
for i = 1:size(dataList, 1)
    d = dataList{i,1};
    all_pixels = [all_pixels; d(:)]; 
end
robust_min = prctile(all_pixels, 0.1); 
robust_max = prctile(all_pixels, 99.8); 
z_lims = [robust_min, robust_max];
clear all_pixels;
%% 3. FIGURE SETTINGS
figWidth  = 17.5;
figHeight = 10;
fontSize  = 10;
fontName  = 'Times New Roman';

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_Images_2D_MM_turbo', ...
             'NumberTitle', 'off');

t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4. DRAW 5 SUBFIGURES (2D)
num_imgs = 6;
cols_fig = 3;     % số cột của layout

labels = {'(a)', '(b)', '(c)', '(d)', '(e)', 'f'};
axs = gobjects(1,6);

for i = 1:6
    axs(i) = nexttile;

    data = dataList{i,1};

    imagesc(x_vec, y_vec, data);
    axis image;
    clim(z_lims);
    colormap(gca, turbo);

    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
        'FontName','Times New Roman', 'Interpreter','latex');
    % Trục X
    xlabel('x (mm)', 'Interpreter', 'latex');
    % Trục Y
    ylabel('y (mm)', 'Interpreter', 'latex');
    % set(gca, 'XTick', [], 'YTick', []);
    set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
        'LineWidth', 1, 'TickLabelInterpreter', 'latex');
    box on;
end
cb = colorbar;
cb.Layout.Tile = 'east'; 
cb.Limits = z_lims;

cb.TickLabelInterpreter = 'latex';
cb.FontSize = fontSize;

cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize + 1;

saveFolder = fullfile(pwd, 'ExportedFigures_simulation');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_Images_2D_MM_turbo' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

%%
% Ảnh màu jet
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_Images_2D_MM_jet', ...
             'NumberTitle', 'off');

t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4. DRAW 5 SUBFIGURES (2D)
num_imgs = 6;
cols_fig = 3;     % số cột của layout

labels = {'(a)', '(b)', '(c)', '(d)', '(e)', 'f'};
axs = gobjects(1,6);

for i = 1:6
    axs(i) = nexttile;

    data = dataList{i,1};

    imagesc(x_vec, y_vec, data);
    axis image;
    clim(z_lims);
    colormap(gca, jet);

    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
        'FontName','Times New Roman', 'Interpreter','latex');
    % Trục X
    xlabel('x (mm)', 'Interpreter', 'latex');
    % Trục Y
    ylabel('y (mm)', 'Interpreter', 'latex');
    % set(gca, 'XTick', [], 'YTick', []);
    set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
        'LineWidth', 1, 'TickLabelInterpreter', 'latex');
    box on;

end
cb = colorbar;
cb.Layout.Tile = 'east'; 
cb.Limits = z_lims;

cb.TickLabelInterpreter = 'latex';
cb.FontSize = fontSize;

cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize + 1;

saveFolder = fullfile(pwd, 'ExportedFigures_simulation');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_Images_2D_MM_jet' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

%% ảnh 3D - turbo
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList{1,1});
x_vec = (0:cols-1) * px_size;
y_vec = (0:rows-1) * px_size;
[X, Y] = meshgrid(x_vec, y_vec);

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_Images_3D_MM_turbo', ...
             'NumberTitle', 'off');

t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

labels = {'(a)','(b)','(c)','(d)','(e)','(f)'};
axs = gobjects(1,6);

for i = 1:6
    axs(i) = nexttile;
    surf(X, Y, dataList{i,1}, 'EdgeColor','none');
    shading flat;
    colormap(gca, turbo);
    clim(z_lims);
    zlim(z_lims);
    
    axis tight; axis vis3d;
    view(3); pbaspect([1 1 0.6]);
    
    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
          'FontName','Times New Roman', 'Interpreter','latex');
    xlabel('x (mm)','Interpreter','latex');
    ylabel('y (mm)','Interpreter','latex');

    set(gca,'FontName',fontName,'FontSize',fontSize,'LineWidth',1,'TickLabelInterpreter','latex');
    box on;
 
end
cb = colorbar;
cb.Layout.Tile = 'east'; 
cb.Limits = z_lims;

cb.TickLabelInterpreter = 'latex';
cb.FontSize = fontSize;

cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize + 1;

saveFolder = fullfile(pwd, 'ExportedFigures_simulation');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_Images_3D_MM_turbo' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

%% anh 3D -jet
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_Images_3D_MM_jet', ...
             'NumberTitle', 'off');

t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

%% 4. DRAW 5 SUBFIGURES 3D
labels = {'(a)','(b)','(c)','(d)','(e)','(f)'};
axs = gobjects(1,6);

for i = 1:6
    axs(i) = nexttile;
    surf(X, Y, dataList{i,1}, 'EdgeColor','none');
    shading flat;
    colormap(gca, jet);
    clim(z_lims);
    zlim(z_lims);
    
    axis tight; axis vis3d;
    view(3); pbaspect([1 1 0.6]);
    
    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
          'FontName','Times New Roman', 'Interpreter','latex');
    % Trục X
    xlabel('x (mm)','Interpreter','latex');
    % Trục Y
    ylabel('y (mm)','Interpreter','latex');

    % set(gca, 'XTick', [], 'YTick', []);
    set(gca,'FontName',fontName,'FontSize',fontSize,'LineWidth',1,'TickLabelInterpreter','latex');
    box on;

end
cb = colorbar;
cb.Layout.Tile = 'east'; 
cb.Limits = z_lims;

cb.TickLabelInterpreter = 'latex';
cb.FontSize = fontSize;

cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize + 1;

saveFolder = fullfile(pwd, 'ExportedFigures_simulation');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_Images_3D_MM_jet' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

%% Hiển thị sai số
error_goldstein = final_phi_goldstein - object_phase_without_noise;
error_quality = final_phi_quality - object_phase_without_noise;
error_tie = final_phi_tie_dct - object_phase_without_noise;
error_wls = final_phi_wls - object_phase_without_noise;
error_proposed = final_phi_proposed - object_phase_without_noise;

error_goldstein = error_goldstein - min(error_goldstein(:));
error_quality = error_quality - min(error_quality(:));
error_tie = error_tie - min(error_tie(:));
error_wls = error_wls - min(error_wls(:));
error_proposed = error_proposed - min(error_proposed(:));

dataList_error = {...
    error_goldstein,   'Goldstein'; ...
    error_quality,     'Quality-Guided'; ...
    error_tie,     'TIE-DCT'; ...
    error_wls,         'WLS'; ...
    error_proposed,    'Proposed (Final)' ...
    };

px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList_error{1,1});
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

%% 2. GLOBAL COLOR LIMITS
all_pixels = []; 
for i = 1:size(dataList_error, 1)
    d = dataList_error{i,1};
    all_pixels = [all_pixels; d(:)]; 
end
robust_min = prctile(all_pixels, 0.1); 
robust_max = prctile(all_pixels, 99.8); 
z_lims = [robust_min, robust_max];
clear all_pixels;

%% 3. FIGURE SETTINGS
figWidth  = 17.5;
figHeight = 10;
fontSize  = 10;
fontName  = 'Times New Roman';
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_Images_2D_MM_sai_so', ...
             'NumberTitle', 'off');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

num_imgs = 5;

labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};
axs = gobjects(1,5);
for i = 1:5
    axs(i) = nexttile;

    data = dataList_error{i,1};

    imagesc(x_vec, y_vec, data);
    axis image;
    clim(z_lims);
    colormap(gca, turbo);

    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
        'FontName','Times New Roman', 'Interpreter','latex');

    xlabel('x (mm)', 'Interpreter', 'latex');
    ylabel('y (mm)', 'Interpreter', 'latex');
    % set(gca, 'XTick', [], 'YTick', []);

    set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
        'LineWidth', 1, 'TickLabelInterpreter', 'latex');
    box on;
end

cb = colorbar;
cb.Layout.Tile = 'east'; 
cb.Limits = z_lims;

cb.TickLabelInterpreter = 'latex';
cb.FontSize = fontSize;

cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize + 1;

saveFolder = fullfile(pwd, 'ExportedFigures_simulation');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
fileName = ['Fig_Comparison_5_Images_2D_MM_sai_so' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector


%%
fprintf("Hoàn thành toàn bộ quá trình.\n");
