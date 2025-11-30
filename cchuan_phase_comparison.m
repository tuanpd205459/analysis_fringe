clc, clear, close all;
%%
tic
load("chuong_trinh_chinh_tao_phase.mat");

%%
% --- Unwrap phase bằng các thuật toán ---
phi_tie_dct      = Unwrap_TIE_DCT_Iter(wrapped_phase);     % TIE với DCT
phi_quality      = unwrap_quality(wrapped_phase);          % Quality-guided
phi_wls          = phase_unwrap_2dweight(wrapped_phase);      % 2D Weighted LS
phi_proposed     = finalUnwrappedPhase;                                     % Proposed / Hybrid
phi_goldstein    = unwrap_goldstein(wrapped_phase);     % goldstein branch-cut

% --- Crop tất cả về cùng kích thước nhỏ nhất ---
[phi_goldstein, phi_tie_dct, phi_quality, phi_wls, phi_proposed] = ...
    crop_multiple_to_smallest(phi_goldstein, phi_tie_dct, phi_quality, phi_wls, phi_proposed);

% --- Kích thước ảnh ---
[M,N] = size(phi_goldstein);
%%


%% 2. PROCESSING PROPOSAL METHOD (Code xử lý của bạn)
% Bước 1: Zernike Removal
coeff = [25, 25]; % Hệ số Zernike

% Gọi hàm của bạn (Giả sử hàm này đã có trong path)
%% 1. ZERNIKE REMOVAL (Loại bỏ quang sai/nghiêng)
% Giả sử 'coeff' và các biến 'phi_...' đầu vào đã có sẵn
% Chuẩn hóa tên biến đầu ra bắt đầu bằng 'final_'

[~, final_phi_proposed]  = ZernikeLegendreFit_removal(phi_proposed, "2indices", coeff);
[~, final_phi_goldstein] = ZernikeLegendreFit_removal(phi_goldstein, "2indices", coeff);
[~, final_phi_tie_dct]   = ZernikeLegendreFit_removal(phi_tie_dct, "2indices", coeff);
[~, final_phi_quality]   = ZernikeLegendreFit_removal(phi_quality, "2indices", coeff);
[~, final_phi_wls]       = ZernikeLegendreFit_removal(phi_wls, "2indices", coeff);

%% 1. PACK RESULTS (ĐÃ BỎ 'PROPOSED RAW')
dataList = { ...
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
g_min = inf; 
g_max = -inf;
for i = 1:size(dataList, 1)
    d = dataList{i,1};
    g_min = min(g_min, min(d(:)));
    g_max = max(g_max, max(d(:)));
end
z_lims = [g_min, g_max];

%% 3. FIGURE SETTINGS
figWidth  = 17.5;
figHeight = 10;
fontSize  = 10;
fontName  = 'Times New Roman';

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4. DRAW 5 SUBFIGURES (2D)
num_imgs = 5;

labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};
axs = gobjects(1,5);

for i = 1:5
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


    set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
             'LineWidth', 1, 'TickLabelInterpreter', 'latex');
    box on;
end

%% 5. COLORBAR (CHUNG)
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize;
cb.TickLabelInterpreter = 'latex';
cb.Limits = z_lims;

%% 6. EXPORT
exportName = 'Fig_Comparison_5_Images_2D_MM';
exportgraphics(fig, [exportName '.png'], 'Resolution', 600);
%%
% Ảnh màu jet
%% 1.5 SETUP SPATIAL COORDINATES (MM)
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList{1,1});
x_vec = (0 : cols-1) * px_size;
y_vec = (0 : rows-1) * px_size;

%% 2. GLOBAL COLOR LIMITS
g_min = inf; 
g_max = -inf;
for i = 1:size(dataList, 1)
    d = dataList{i,1};
    g_min = min(g_min, min(d(:)));
    g_max = max(g_max, max(d(:)));
end
z_lims = [g_min, g_max];

%% 3. FIGURE SETTINGS
figWidth  = 17.5;
figHeight = 10;
fontSize  = 10;
fontName  = 'Times New Roman';

fig = figure('Units', 'centimeters', ...
             'Position', [2, 2, figWidth, figHeight], ...
             'Color', 'w');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4. DRAW 5 SUBFIGURES (2D)
num_imgs = 5;

labels = {'(a)', '(b)', '(c)', '(d)', '(e)'};
axs = gobjects(1,5);

for i = 1:5
    axs(i) = nexttile;

    data = dataList{i,1};

    imagesc(x_vec, y_vec, data);
    axis image;
    clim(z_lims);
    colormap(gca, "jet");

 % Thêm nhãn (a), (b), ...
    title(labels{i}, 'FontWeight','bold', 'FontSize', fontSize+1, ...
          'FontName','Times New Roman', 'Interpreter','latex');

    % Trục X
        xlabel('x (mm)', 'Interpreter', 'latex');


    % Trục Y
        ylabel('y (mm)', 'Interpreter', 'latex');


    set(gca, 'FontName', fontName, 'FontSize', fontSize, ...
             'LineWidth', 1, 'TickLabelInterpreter', 'latex');
    box on;
end

%% 5. COLORBAR (CHUNG)
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize;
cb.TickLabelInterpreter = 'latex';
cb.Limits = z_lims;

%% 6. EXPORT
exportName = 'Fig_Comparison_5_Images_2D_MM_jet';
exportgraphics(fig, [exportName '.png'], 'Resolution', 600);

%%


% ảnh 3D
%% 1. PACK RESULTS
dataList = { ...
    final_phi_goldstein,   'Goldstein'; ...
    final_phi_quality,     'Quality-Guided'; ...
    final_phi_tie_dct,     'TIE-DCT'; ...
    final_phi_wls,         'WLS'; ...
    final_phi_proposed,    'Proposed (Final)' ...
};

%% 1.5 SETUP SPATIAL COORDINATES (MM)
px_size = 3.45e-3; % 3.45 µm = 0.00345 mm
[rows, cols] = size(dataList{1,1});
x_vec = (0:cols-1) * px_size;
y_vec = (0:rows-1) * px_size;
[X, Y] = meshgrid(x_vec, y_vec);

%% 2. GLOBAL COLOR LIMITS
g_min = inf; g_max = -inf;
for i = 1:size(dataList,1)
    d = dataList{i,1};
    g_min = min(g_min, min(d(:)));
    g_max = max(g_max, max(d(:)));
end
z_lims = [g_min, g_max];

%% 3. FIGURE SETTINGS
figWidth = 17.5; figHeight = 10;
fontSize = 10; fontName = 'Times New Roman';

fig = figure('Units','centimeters','Position',[2,2,figWidth,figHeight],'Color','w');
t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

%% 4. DRAW 5 SUBFIGURES 3D
labels = {'(a)','(b)','(c)','(d)','(e)'};
axs = gobjects(1,5);

for i = 1:5
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

    % Trục X
        xlabel('x (mm)','Interpreter','latex');

    % Trục Y
        ylabel('y (mm)','Interpreter','latex');


    set(gca,'FontName',fontName,'FontSize',fontSize,'LineWidth',1,'TickLabelInterpreter','latex');
    box on;
end

%% 5. COLORBAR (CHUNG)
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize;
cb.TickLabelInterpreter = 'latex';
cb.Limits = z_lims;

%% 6. EXPORT
exportName = 'Fig_Comparison_5_Images_3D_MM_labels';
exportgraphics(fig, [exportName '.png'], 'Resolution', 600);


%% ảnh màu 3D-jet
fig = figure('Units','centimeters','Position',[2,2,figWidth,figHeight],'Color','w');
t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

%% 4. DRAW 5 SUBFIGURES 3D
labels = {'(a)','(b)','(c)','(d)','(e)'};
axs = gobjects(1,5);

for i = 1:5
    axs(i) = nexttile;
    
    surf(X, Y, dataList{i,1}, 'EdgeColor','none');
    shading flat;
    colormap(gca, "jet");
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


    set(gca,'FontName',fontName,'FontSize',fontSize,'LineWidth',1,'TickLabelInterpreter','latex');
    box on;
end

%% 5. COLORBAR (CHUNG)
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Phase (rad)';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize = fontSize;
cb.TickLabelInterpreter = 'latex';
cb.Limits = z_lims;

%% 6. EXPORT
exportName = 'Fig_Comparison_5_Images_3D_MM_labels_jet';
exportgraphics(fig, [exportName '.png'], 'Resolution', 600);




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
