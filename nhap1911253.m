%% --- GIAI ĐOẠN 1: TIỀN XỬ LÝ NÂNG CAO CHO VÂN MỜ ---
clc, clear,close all;

%%
baseFolder = "C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 12 11 25\60x o thu 6\251112\anh oke";
fileName = "image_2025-11-12T18-33-12.732.bmp";

% Thông số xử lý ảnh
SENSITIVITY_COEF = 0.65;   % Hệ số nhạy adaptive threshold
NEIGHBORHOOD_SIZE = 35;    % Kích thước vùng lân cận (phải là số lẻ)
MIN_BRANCH_LENGTH = 5;    % Ngưỡng độ dài nhánh thừa cần xóa (thay cho distThresh cũ)
GAUSS_SIGMA = 1;           % Độ làm mượt ảnh

%% --- 2. ĐỌC VÀ TIỀN XỬ LÝ ẢNH ---
imgPath = fullfile(baseFolder, fileName);

if ~isfile(imgPath)
    % Fallback: Nếu không tìm thấy file cứng, mở hộp thoại chọn
    fprintf('Không tìm thấy file mặc định. Vui lòng chọn ảnh...\n');
    [fileName, baseFolder] = uigetfile({'*.bmp;*.png;*.jpg'}, 'Chọn file ảnh');
    if isequal(fileName, 0), return; end
    imgPath = fullfile(baseFolder, fileName);
end

hologram = imread(imgPath);
if size(hologram, 3) == 3
    hologram = rgb2gray(hologram);
end

% Xoay ảnh (Lưu ý: rot90(img, k). k=1 là 90 độ, k=2 là 180 độ)
% Theo comment cũ của bạn là 180 độ, nên tôi để k=2. Nếu muốn 90 thì sửa thành 1.
hologram = rot90(hologram, 1); 
%%
% 1. Làm sắc nét ảnh (Sharpening)
% Radius: độ rộng của biên cần làm nét (2-3 px cho vân dày)
% Amount: độ mạnh (1-2 là vừa, cao quá sẽ nhiễu)
hologram_sharp = imsharpen(hologram, 'Radius', 2, 'Amount', 3);

% 2. Lọc nhiễu đốm mà vẫn giữ biên (Median Filter tốt hơn Gaussian cho việc này)
hologram_denoised = medfilt2(hologram_sharp, [5 5]); 

% 3. QUAN TRỌNG: Tăng cường cấu trúc dạng sợi (Ridge Enhancement)
% 'ObjectPolarity': 'dark' nếu vân màu đen nền trắng, 'bright' nếu vân trắng nền đen.
% Dựa trên ảnh của bạn (hologram), tôi đoán là vân tối màu trên nền sáng -> 'dark'
% Nếu ngược lại, hãy đổi thành 'bright'
hologram_ridge = fibermetric(hologram_denoised, ...
    'StructureSensitivity', 1, ... % Độ nhạy: càng thấp càng bắt được vân mờ (nhưng dễ bắt nhiễu)
    'ObjectPolarity', 'bright');      

% Đảo ngược lại ảnh ridge (để vân thành màu trắng, nền đen cho dễ xử lý sau này)
% hologram_ridge = imcomplement(hologram_ridge); 

%% --- GIAI ĐOẠN 2: NHỊ PHÂN HÓA ---
% Lúc này ảnh hologram_ridge đã rất rõ nét, ta có thể dùng ngưỡng đơn giản hơn
% hoặc tiếp tục dùng adaptive.

% Normalize về 0-1
hologram_final = mat2gray(hologram_ridge);

% Nhị phân hóa (Giảm Sensitivity xuống thấp một chút để bắt được cả nét mờ)
T = adaptthresh(hologram_final, 0.4, 'NeighborhoodSize', [51 51], 'Statistic', 'mean');
BW = imbinarize(hologram_final, T);

hologram_bina = BW;
%% --- GIAI ĐOẠN 3: XỬ LÝ HÌNH THÁI & SKELETON ---
% 1. Nối các điểm đứt gãy do mờ (Bridge)
BW = bwmorph(BW, 'bridge', Inf); 

% 2. Đóng khe hở và lấp lỗ (như code trước)
% BW = imclose(BW, strel('disk', 2));
BW = imfill(BW, 'holes');
% BW = bwareaopen(BW, 50); 
BW = bwmorph(BW, 'majority', Inf); % Làm mượt biên
% 3. Skeletonize
min_branch_len = 5;
BW_skel = bwskel(BW, 'MinBranchLength', min_branch_len);

%% --- HIỂN THỊ ---
figure('Name', 'Xử lý Vân Mờ', 'Color', 'w');
t = tiledlayout(2, 3, 'TileSpacing', 'compact');

nexttile; imshow(hologram); title('1. Ảnh gốc (có chỗ mờ)');
figure; imshow(hologram_sharp); title('2. Sharpening');
figure; imshow(hologram_ridge); title('3. Fibermetric (Ridge detect)');
figure; imshow(BW); title('4. Binary (Đã nối)');
figure; imshowpair(BW_skel, hologram_bina, 'falsecolor'); title('5. Overlay');
figure; imshow(BW_skel); title('6. Final Skeleton');