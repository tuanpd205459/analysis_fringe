%% 1. INITIALIZATION & DATA LOADING
clc; clear; close all;

% Load dữ liệu
% Đảm bảo file .mat chứa các biến: 
% unwrapped_Phase_2dweight, unwrapped_Phase_LS_DCT, 
% unwrapped_Phase_noncontinue, unwrapped_Phase_proposal, unwrapped_Phase_TIE_FFT
load("chuong_trinh_chinh_anh_that.mat"); 

% --- Cấu hình chung cho bài báo ---
figWidth = 18;  % cm (Full width cho Optics Express là khoảng 17-18cm)
figHeight = 10; % cm
fontName = 'Times New Roman';
fontSize = 10;
lineWidth = 1.0;

%% 2. PROCESSING PROPOSAL METHOD (Code xử lý của bạn)
% Bước 1: Zernike Removal
z_map = unwrapped_Phase_proposal; % Biến gốc từ Proposal
coeff = [25, 25]; % Hệ số Zernike

% Gọi hàm của bạn (Giả sử hàm này đã có trong path)
% Nếu chưa có hàm, bạn cần addpath hoặc tạo hàm giả lập
try
    [~, z_recon_map2] = ZernikeLegendreFit_removal(z_map, "2indices", coeff);
catch
    warning('Không tìm thấy hàm ZernikeLegendreFit_removal. Đang dùng dữ liệu gốc.');
    z_recon_map2 = z_map; 
end

% Bước 2: Smoothing (Lọc mượt) - Chọn Cách 2 (Gaussian) như bạn note
sigma = 2; 
window = 6 * sigma;
h = fspecial('gaussian', window, sigma);
final_proposal_smoothed = imfilter(z_recon_map2, h, 'replicate');

%% 3. PREPARE DATA FOR PLOTTING
% Gom dữ liệu vào Cell Array để vẽ vòng lặp cho gọn
dataList = { ...
    unwrapped_Phase_2dweight, '2D Weight'; ...
    unwrapped_Phase_LS_DCT, 'LS DCT'; ...
    unwrapped_Phase_noncontinue, 'Non-continuous'; ...
    unwrapped_Phase_TIE_FFT, 'TIE FFT'; ...
    unwrapped_Phase_proposal, 'Proposal (Raw)'; ...
    final_proposal_smoothed, 'Proposal (Corrected)' ... % Kết quả cuối cùng
};

% Tự động tính giới hạn trục Z (Color limit) dựa trên ảnh kết quả tốt nhất
% Điều này giúp so sánh công bằng độ phẳng của các phương pháp
ref_img = final_proposal_smoothed;
z_lims = [min(ref_img(:)), max(ref_img(:))]; 
% Hoặc nếu muốn auto cho từng ảnh thì set z_lims = [];

%% 4. VISUALIZATION (Tiled Layout)
fig = figure('Units', 'centimeters', 'Position', [2, 2, figWidth, figHeight]);
set(fig, 'Color', 'w');

% Tạo lưới 2 hàng x 3 cột
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:6
    nexttile;
    currentData = dataList{i, 1};
    titleStr = dataList{i, 2};
    
    % Vẽ 3D Surface
    s = surf(currentData, 'EdgeColor', 'none'); 
    
    % Tối ưu hiển thị
    axis tight; 
    view(3); % Góc nhìn 3D mặc định
    % view(-37.5, 30); % Góc nhìn tùy chỉnh nếu cần xoay
    
    % Ánh sáng và đổ bóng (Làm khối nổi bật hơn - chuẩn OE hay dùng)
    camlight; lighting gouraud; shading interp;
    
    % Colormap
    % Khuyên dùng 'turbo' thay vì 'jet' cho bài báo hiện đại (độ tương phản tốt hơn)
    colormap(gca, turbo); 
    
    % Đồng bộ trục Z nếu cần so sánh độ lớn
    % caxis(z_lims); 
    
    % Trang trí trục
    title(['\textbf{(' char(96+i) ') ' titleStr '}'], 'Interpreter', 'latex', 'FontSize', fontSize);
    
    % Chỉ hiện nhãn trục ở các hình biên để đỡ rối (Tùy chọn)
    if i > 3; xlabel('x (pixel)', 'Interpreter', 'latex'); end
    if mod(i,3)==1; ylabel('y (pixel)', 'Interpreter', 'latex'); end
    
    set(gca, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', lineWidth);
    box on;
end

% Thêm Colorbar chung (Shared Colorbar)
c = colorbar;
c.Layout.Tile = 'east'; % Đặt bên phải ngoài cùng
c.Label.String = 'Phase (rad)';
c.Label.Interpreter = 'latex';
c.Label.FontSize = fontSize;
c.TickLabelInterpreter = 'latex';

%% 5. EXPORT
exportName = 'Fig_Comparison_Methods';
% Xuất file ảnh PNG độ phân giải cao (600 DPI)
exportgraphics(fig, [exportName '.png'], 'Resolution', 600);
% Xuất file PDF Vector (cho vào LaTeX/Word không bị vỡ)
exportgraphics(fig, [exportName '.pdf'], 'ContentType', 'vector');

disp('Đã xuất hình ảnh thành công!');