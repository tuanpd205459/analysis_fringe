clc, clear, close all;

%% --- 1. CẤU HÌNH THAM SỐ (PARAMETERS) ---
% Thông số đường dẫn
baseFolder = "C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 12 11 25\60x o thu 6\251112\anh oke";
fileName = "image_2025-11-12T18-33-12.732.bmp";

% Thông số xử lý ảnh
SENSITIVITY_COEF = 0.65;   % Hệ số nhạy adaptive threshold
NEIGHBORHOOD_SIZE = 51;    % Kích thước vùng lân cận (phải là số lẻ)
MIN_BRANCH_LENGTH = 8;    % Ngưỡng độ dài nhánh thừa cần xóa (thay cho distThresh cũ)
GAUSS_SIGMA = 1;           % Độ làm mượt ảnh
distThresh = 6;
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

% 2.1. Lọc nhiễu và cân bằng histogram
hologram = imgaussfilt(hologram, GAUSS_SIGMA);

hologram = adapthisteq(hologram);

%% --- 3. NHỊ PHÂN HÓA (BINARIZATION) ---
% Tính ngưỡng thích nghi
T = adaptthresh(hologram, SENSITIVITY_COEF, ...
    'NeighborhoodSize', [NEIGHBORHOOD_SIZE NEIGHBORHOOD_SIZE], ...
    'Statistic', 'median');

BW = imbinarize(hologram, T);

% Xử lý hình thái học (Morphological) để làm mịn biên và nối liền

% Thicken -> Close -> Fill holes
BW = bwmorph(BW, "thicken", 2);
% Tạo "đầu bút" hình tròn bán kính 2 pixel (tăng lên nếu khe hở lớn)
se = strel('disk', 1); 

% Thực hiện đóng (Nối đứt gãy)
BW = imclose(BW, se);
figure; 
imshow(BW); 
title('5. bw sau khi imclose');



%% --- 4. SKELETONIZE & LÀM SẠCH (TỐI ƯU HÓA) ---
fprintf('Đang thực hiện Skeletonize và xóa nhánh nhỏ...\n');

% --- CÁCH MỚI (Tối ưu tốc độ và hiệu quả) ---
% bwskel: Hàm này nhanh hơn và mạnh hơn bwmorph.
% 'MinBranchLength': Tự động xóa các nhánh cụt ngắn hơn giá trị này (thay thế vòng lặp geodesic)
BW_skel_clean = bwskel(BW, 'MinBranchLength', MIN_BRANCH_LENGTH);

% Làm sạch thêm các điểm cô lập (nếu cần)
BW_skel_clean = bwmorph(BW_skel_clean, 'clean');

%% --- 5. HIỂN THỊ KẾT QUẢ (VISUALIZATION) ---
% Hình 5: So sánh chồng ảnh (Overlay)
figure; 
imshowpair(BW_skel_clean, BW, 'falsecolor'); 
title('5. Skeleton Overlay');

% Hình 6: Kết quả Skeleton cuối cùng
figure(); imshow(BW_skel_clean); 
title(['6. Final Skeleton (Đã xóa nhánh < ', num2str(MIN_BRANCH_LENGTH), 'px)']);

%%
branchpoints = bwmorph(BW_skel_clean, 'branchpoints');
[rows, cols] = find(branchpoints); 
endpoints = bwmorph(BW_skel_clean,"endpoints");
[r, c ] = find(endpoints);
figure;
imshow(BW_skel_clean);
hold on;
plot(cols, rows, "go", 'MarkerSize', 10, 'LineWidth',1.5);
plot(c, r, "r+", 'MarkerSize', 10, 'LineWidth',1.5);
hold off;
%%








%% --- 1. CẤU HÌNH THAM SỐ (PARAMETERS) ---
% Thông số đường dẫn
baseFolder = "C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 12 11 25\60x o thu 6\251112\anh oke";
fileName = "image_2025-11-12T18-33-12.732.bmp";

% Thông số xử lý ảnh
SENSITIVITY_COEF = 0.65;   % Hệ số nhạy adaptive threshold
NEIGHBORHOOD_SIZE = 51;    % Kích thước vùng lân cận (phải là số lẻ)
MIN_BRANCH_LENGTH = 8;    % Ngưỡng độ dài nhánh thừa cần xóa (thay cho distThresh cũ)
GAUSS_SIGMA = 1;           % Độ làm mượt ảnh
distThresh = 6;
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

% 2.1. Lọc nhiễu và cân bằng histogram
hologram = imgaussfilt(hologram, GAUSS_SIGMA);

hologram = adapthisteq(hologram);

%% --- 3. NHỊ PHÂN HÓA (BINARIZATION) ---
% Tính ngưỡng thích nghi
T = adaptthresh(hologram, SENSITIVITY_COEF, ...
    'NeighborhoodSize', [NEIGHBORHOOD_SIZE NEIGHBORHOOD_SIZE], ...
    'Statistic', 'median');

BW = imbinarize(hologram, T);
fprintf('Đang thực hiện Skeletonize và xóa nhánh nhỏ...\n');

se_cut = strel('disk', 1); 
BW_open = imopen(BW, se_cut);

figure; 
imshow(BW_open); 
title(' anh bw sau khi imopen');

% --- CÁCH MỚI (Tối ưu tốc độ và hiệu quả) ---
% bwskel: Hàm này nhanh hơn và mạnh hơn bwmorph.
% 'MinBranchLength': Tự động xóa các nhánh cụt ngắn hơn giá trị này (thay thế vòng lặp geodesic)
BW_skel_clean = bwskel(BW_open, 'MinBranchLength', MIN_BRANCH_LENGTH);

% Làm sạch thêm các điểm cô lập (nếu cần)
BW_skel_clean = bwmorph(BW_skel_clean, 'clean');

%% --- 5. HIỂN THỊ KẾT QUẢ (VISUALIZATION) ---
% Hình 5: So sánh chồng ảnh (Overlay)
figure; 
imshowpair(BW_skel_clean, BW_open, 'falsecolor'); 
title('5. Skeleton Overlay');

% Hình 6: Kết quả Skeleton cuối cùng
figure(); imshow(BW_skel_clean); 
title(['6. Final Skeleton (Đã xóa nhánh < ', num2str(MIN_BRANCH_LENGTH), 'px)']);
%%
branchpoints = bwmorph(BW_skel_clean, 'branchpoints');

[rows, cols] = find(branchpoints); 
endpoints = bwmorph(BW_skel_clean, "endpoints");
[r,c] = find(endpoints);

figure;
imshow(BW_skel_clean);
hold on;
plot(cols, rows, "go", 'MarkerSize', 10, 'LineWidth',1.5);
plot(c, r, "r+", 'MarkerSize', 10, 'LineWidth',1.5);

hold off;

