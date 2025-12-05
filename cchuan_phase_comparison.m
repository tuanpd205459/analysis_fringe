clc, clear, close all;
%%
tic
load("cchuong_trinh_chinh_tao_phase.mat");
addpath("D:\tuan\analysis\analysis_fringe\export_fig");

%%
% --- Unwrap phase bằng các thuật toán ---
phi_tie_dct      = unwrap_TIE_FD_DCT(wrapped_phase);     % TIE với DCT
phi_quality      = unwrap_quality(wrapped_phase);          % Quality-guided
phi_wls          = phase_unwrap_2dweight(wrapped_phase);      % 2D Weighted LS
phi_proposed     = finalUnwrappedPhase;                                     % Proposed / Hybrid
phi_goldstein    = unwrap_goldstein(wrapped_phase);     % goldstein branch-cut

% --- Crop tất cả về cùng kích thước nhỏ nhất ---
[wrapped_phase, phi_goldstein, phi_tie_dct, phi_quality, phi_wls, phi_proposed] = ...
    crop_multiple_to_smallest(wrapped_phase, phi_goldstein, phi_tie_dct, phi_quality, phi_wls, phi_proposed);

[M,N] = size(phi_goldstein);

%% 2. PROCESSING PROPOSAL METHOD (Code xử lý của bạn)
coeff = [25, 25]; % Hệ số Zernike

% 1. ZERNIKE REMOVAL (Loại bỏ quang sai/nghiêng)
[~, final_phi_proposed]  = ZernikeLegendreFit_removal(phi_proposed, "2indices", coeff);
[~, final_phi_goldstein] = ZernikeLegendreFit_removal(phi_goldstein, "2indices", coeff);
[~, final_phi_tie_dct]   = ZernikeLegendreFit_removal(phi_tie_dct, "2indices", coeff);
[~, final_phi_quality]   = ZernikeLegendreFit_removal(phi_quality, "2indices", coeff);
[~, final_phi_wls]       = ZernikeLegendreFit_removal(phi_wls, "2indices", coeff);

dataList = { ...
    wrapped_phase,              'wrapped_phase';...
    final_phi_goldstein,         'Goldstein'; ...
    final_phi_quality,           'Quality-Guided'; ...
    final_phi_tie_dct,           'TIE-DCT'; ...
    final_phi_wls,               'WLS'; ...
    final_phi_proposed,          'Proposed (Final)' ...
};

% 1.5 SETUP SPATIAL COORDINATES (MM)
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList{1,1});
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;
%
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
             'Name', 'Fig_Comparison_5_Real_2D_MM_turbo', ...
             'NumberTitle', 'off');

t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4. DRAW 5 SUBFIGURES (2D)
num_imgs = length(dataList);
cols_fig = 3;     % số cột của layout

labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};
axs = gobjects(1, length(dataList));

for i = 1: length(dataList)
    axs(i) = nexttile;

    data = dataList{i,1};

    imagesc(x_vec, y_vec, data);
    axis image;
    clim(z_lims);
    colormap(gca, turbo);

    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
        'FontName','Times New Roman', 'Interpreter','latex');

    xlabel('x (mm)', 'Interpreter', 'latex');
    ylabel('y (mm)', 'Interpreter', 'latex');

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

saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_Real_2D_MM_turbo' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

%%
% Ảnh màu jet
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_Real_2D_MM_turbo', ...
             'NumberTitle', 'off');

t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4. DRAW 5 SUBFIGURES (2D)
num_imgs = length(dataList);
cols_fig = 3;     % số cột của layout

labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};
axs = gobjects(1, length(dataList));

for i = 1: length(dataList)
    axs(i) = nexttile;

    data = dataList{i,1};

    imagesc(x_vec, y_vec, data);
    axis image;
    clim(z_lims);
    colormap(gca, "jet");

    % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
        'FontName','Times New Roman', 'Interpreter','latex');

    xlabel('x (mm)', 'Interpreter', 'latex');
    ylabel('y (mm)', 'Interpreter', 'latex');

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

saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_Real_2D_MM_jet' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector


%% ảnh 3D
dataList = { ...
    wrapped_phase,              'wrapped_phase';...

    final_phi_goldstein,         'Goldstein'; ...
    final_phi_quality,           'Quality-Guided'; ...
    final_phi_tie_dct,           'TIE-DCT'; ...
    final_phi_wls,               'WLS'; ...
    final_phi_proposed,          'Proposed (Final)' ...
};

px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList{1,1});
x_vec = (0:cols-1) * px_size;
y_vec = (0:rows-1) * px_size;
[X, Y] = meshgrid(x_vec, y_vec);

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_real_3D_MM_turbo', ...
             'NumberTitle', 'off');

t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

labels = {'(a)','(b)','(c)','(d)','(e)','(f)'};
axs = gobjects(1, length(dataList));

for i = 1: length(dataList)
    axs(i) = nexttile;
    surf(X, Y, dataList{i,1}, 'EdgeColor','none');
    shading flat;
    colormap(gca, turbo);
    clim(z_lims);
    zlim(z_lims);
    
    % Cấu hình trục và góc nhìn
    axis tight; 
    axis vis3d;           % Giữ tỉ lệ khi xoay
    view(-45, 30);        % [QUAN TRỌNG] Cố định góc nhìn để chữ xoay đúng hướng
    pbaspect([1 1 0.6]);  % Tỉ lệ hộp
    
    % Trục X
    hx = xlabel('$x$ (mm)', 'Interpreter', 'latex');
    set(hx, 'Rotation', 30);                % Xoay nghiêng (chỉnh số này nếu cần)
    set(hx, 'VerticalAlignment', 'middle'); 
    set(hx, 'HorizontalAlignment', 'left'); % Căn lề trái để bám theo trục
    
    % Trục Y
    hy = ylabel('$y$ (mm)', 'Interpreter', 'latex');
    set(hy, 'Rotation', -25);               % Xoay nghiêng ngược lại
    set(hy, 'VerticalAlignment', 'middle');
    set(hy, 'HorizontalAlignment', 'right'); % Căn lề phải để bám theo trục

    % --- 3. XỬ LÝ TRỤC Z (MỚI THÊM) ---
    % Bạn thay đổi nội dung '$z$ (mm)' thành đơn vị thực tế (ví dụ: 'Phase (rad)')
    hz = zlabel('phase (rad)', 'Interpreter', 'latex'); 
    
    % Mẹo xử lý trục Z:
    set(hz, 'Rotation', 90); % Xoay 90 độ để chạy dọc theo trục đứng
    % set(hz, 'Rotation', 0); % Hoặc để 0 nếu muốn chữ nằm ngang dễ đọc hơn
    
    % Đẩy chữ Z ra xa trục một chút để không đè lên số (Quan trọng)
    set(hz, 'Units', 'normalized'); % Chuyển đơn vị về 0-1 để dễ chỉnh
    % Lấy vị trí hiện tại
    z_pos = get(hz, 'Position');    
    % Dịch sang trái một chút (giá trị âm ở phần tử đầu tiên)
    set(hz, 'Position', z_pos + [-0.0 0 0]);


    title(labels{i}, 'FontWeight','normal', 'FontSize', fontSize, ...
          'FontName', fontName, 'Interpreter', 'latex');
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

saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_reals_3D_MM_turbo' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

%% anh 3D -jet
fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w', ...
             'Name', 'Fig_Comparison_5_real_3D_MM_jet', ...
             'NumberTitle', 'off');

t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

%% 4. DRAW 5 SUBFIGURES 3D
labels = {'(a)','(b)','(c)','(d)','(e)','(f)'};
axs = gobjects(1, length(dataList));

for i = 1: length(dataList)
    axs(i) = nexttile;
    surf(X, Y, dataList{i,1}, 'EdgeColor','none');
    shading flat;
    colormap(gca, turbo);
    clim(z_lims);
    zlim(z_lims);
    
    % Cấu hình trục và góc nhìn
    axis tight; 
    axis vis3d;           % Giữ tỉ lệ khi xoay
    view(-45, 30);        % [QUAN TRỌNG] Cố định góc nhìn để chữ xoay đúng hướng
    pbaspect([1 1 0.6]);  % Tỉ lệ hộp
    
    % Trục X
    hx = xlabel('$x$ (mm)', 'Interpreter', 'latex');
    set(hx, 'Rotation', 30);                % Xoay nghiêng (chỉnh số này nếu cần)
    set(hx, 'VerticalAlignment', 'middle'); 
    set(hx, 'HorizontalAlignment', 'left'); % Căn lề trái để bám theo trục
    
    % Trục Y
    hy = ylabel('$y$ (mm)', 'Interpreter', 'latex');
    set(hy, 'Rotation', -25);               % Xoay nghiêng ngược lại
    set(hy, 'VerticalAlignment', 'middle');
    set(hy, 'HorizontalAlignment', 'right'); % Căn lề phải để bám theo trục

    % --- 3. XỬ LÝ TRỤC Z (MỚI THÊM) ---
    % Bạn thay đổi nội dung '$z$ (mm)' thành đơn vị thực tế (ví dụ: 'Phase (rad)')
    hz = zlabel('phase (rad)', 'Interpreter', 'latex'); 
    
    % Mẹo xử lý trục Z:
    set(hz, 'Rotation', 90); % Xoay 90 độ để chạy dọc theo trục đứng
    % set(hz, 'Rotation', 0); % Hoặc để 0 nếu muốn chữ nằm ngang dễ đọc hơn
    
    % Đẩy chữ Z ra xa trục một chút để không đè lên số (Quan trọng)
    set(hz, 'Units', 'normalized'); % Chuyển đơn vị về 0-1 để dễ chỉnh
    % Lấy vị trí hiện tại
    z_pos = get(hz, 'Position');    
    % Dịch sang trái một chút (giá trị âm ở phần tử đầu tiên)
    set(hz, 'Position', z_pos + [-0.0 0 0]);


    title(labels{i}, 'FontWeight','normal', 'FontSize', fontSize, ...
          'FontName', fontName, 'Interpreter', 'latex');
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

saveFolder = fullfile(pwd, 'ExportedFigures_experiments');
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

fileName = ['Fig_Comparison_5_real_3D_MM_jet' timestamp];   % đổi ten anh
fullPath = fullfile(saveFolder, fileName);
export_fig([fullPath '.png'], '-png', '-r600');       % PNG 600 dpi
export_fig([fullPath '.eps'], '-eps', '-opengl');   % EPS vector

save("cchuan_phase_comparison.mat");
toc
%%
function varargout = crop_multiple_to_smallest(varargin)
    % Giả định tất cả các biến là 2D ma trận
    n = nargin;
    sizes = cellfun(@size, varargin, 'UniformOutput', false);

    % Tìm kích thước nhỏ nhất theo từng chiều
    min_rows = min(cellfun(@(s) s(1), sizes));
    min_cols = min(cellfun(@(s) s(2), sizes));

    varargout = cell(1, n);
    for i = 1:n
        mat = varargin{i};
        [m, n_] = size(mat);
        
        % Tính chỉ số cắt đều 4 phía
        row_start = floor((m - min_rows)/2) + 1;
        col_start = floor((n_ - min_cols)/2) + 1;
        row_end = row_start + min_rows - 1;
        col_end = col_start + min_cols - 1;
        
        varargout{i} = mat(row_start:row_end, col_start:col_end);
    end
end
