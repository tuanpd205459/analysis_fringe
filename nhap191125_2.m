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
hologram_denoised = imgaussfilt(hologram, GAUSS_SIGMA);
hologram_eq = adapthisteq(hologram_denoised);

%% --- 3. NHỊ PHÂN HÓA (BINARIZATION) ---
% Tính ngưỡng thích nghi
T = adaptthresh(hologram_eq, SENSITIVITY_COEF, ...
    'NeighborhoodSize', [NEIGHBORHOOD_SIZE NEIGHBORHOOD_SIZE], ...
    'Statistic', 'median');

hologram_bin = imbinarize(hologram_eq, T);

% Xử lý hình thái học (Morphological) để làm mịn biên và nối liền

% Thicken -> Close -> Fill holes
BW_processed = bwmorph(hologram_bin, "thicken", 2);
% Tạo "đầu bút" hình tròn bán kính 2 pixel (tăng lên nếu khe hở lớn)
se = strel('disk', 1); 

% Thực hiện đóng (Nối đứt gãy)
BW_processed = imclose(BW_processed, se);
figure; 
imshow(BW_processed); 
title('5. Skeleton Overlay');

% BW_processed = bwmorph(BW_processed, "close");
% BW_processed = imfill(BW_processed, "holes");

%% --- 4. SKELETONIZE & LÀM SẠCH (TỐI ƯU HÓA) ---
fprintf('Đang thực hiện Skeletonize và xóa nhánh nhỏ...\n');

% --- CÁCH MỚI (Tối ưu tốc độ và hiệu quả) ---
% bwskel: Hàm này nhanh hơn và mạnh hơn bwmorph.
% 'MinBranchLength': Tự động xóa các nhánh cụt ngắn hơn giá trị này (thay thế vòng lặp geodesic)
BW_skel_clean = bwskel(BW_processed, 'MinBranchLength', MIN_BRANCH_LENGTH);

% Làm sạch thêm các điểm cô lập (nếu cần)
BW_skel_clean = bwmorph(BW_skel_clean, 'clean');

%% --- 5. HIỂN THỊ KẾT QUẢ (VISUALIZATION) ---
% Sử dụng tiledlayout để hiển thị gọn gàng trên 1 cửa sổ
figure('Name', 'Quy trình Xử lý Hologram', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
t = tiledlayout(2, 3, 'TileSpacing', 'compact');

% Hình 1: Ảnh gốc (Grayscale)
nexttile; imshow(hologram); title('1. Grayscale Gốc');

% Hình 2: Sau khi lọc nhiễu & EQ
nexttile; imshow(hologram_eq); title('2. Denoise & AdaptHistEq');

% Hình 3: Nhị phân hóa
nexttile; imshow(hologram_bin); title('3. Adaptive Threshold');

% Hình 4: Sau xử lý hình thái học (Trước khi skeleton)
nexttile; imshow(BW_processed); title('4. Morphological Processed');

% Hình 5: So sánh chồng ảnh (Overlay)
figure; 
imshowpair(BW_skel_clean, BW_processed, 'falsecolor'); 
title('5. Skeleton Overlay');

% Hình 6: Kết quả Skeleton cuối cùng
figure(); imshow(BW_skel_clean); 
title(['6. Final Skeleton (Đã xóa nhánh < ', num2str(MIN_BRANCH_LENGTH), 'px)']);

%%
%% --- XỬ LÝ NÂNG CAO: CẮT CẦU NỐI GIẢ (H-Bridges) ---

BW_skel = BW_skel_clean;
% 1. Xác định các điểm ngã ba/ngã tư (Branch Points)
branchPoints = bwmorph(BW_skel, 'branchpoints');

% 2. Xóa các điểm giao này để skeleton vỡ ra thành từng đoạn
% Lưu ý: Chỉ xóa đúng pixel branchpoint, không dilate để bảo toàn cấu trúc
branchPoints_thick = imdilate(branchPoints, ones(3)); 
BW_broken = BW_skel & ~branchPoints_thick;

% 3. Xóa các đoạn quá ngắn (Cầu nối giả)
% BRIDGE_LIMIT: Độ dài tối đa của đoạn bị coi là cầu nối (pixel)
BRIDGE_LIMIT = 15; 
BW_cleaned_parts = bwareaopen(BW_broken, BRIDGE_LIMIT);

% 4. Khôi phục lại Branch Points (Hàn lại)
% Logic: Cộng lại điểm branchPoints vào ảnh đã lọc
BW_restored = BW_cleaned_parts | branchPoints;

% 5. Hậu xử lý (Quan trọng)
% Khi cộng lại, sẽ có trường hợp:
% - Trường hợp 1 (Tốt): Điểm BP lấp vào chỗ trống của vân dài -> Vân liền mạch.
% - Trường hợp 2 (Rác): Cầu nối bị xóa hết, điểm BP được khôi phục nằm trơ trọi 1 mình.
BW_final_skel = bwmorph(BW_restored, 'clean'); % Xóa các điểm pixel cô lập (trường hợp 2)

% Đảm bảo skeleton vẫn mỏng 1 pixel (phòng hờ)
BW_final_skel = bwmorph(BW_final_skel, 'thin', Inf);

%% --- HIỂN THỊ KIỂM TRA ---
figure('Name', 'So sánh Cắt Cầu Nối', 'Color', 'w');
t = tiledlayout(2, 2, 'TileSpacing', 'compact');

nexttile; 
imshow(BW_skel); 
title('1. Skeleton gốc (Có cầu H)');

nexttile; 
imshow(BW_broken); 
title('2. Đã cắt Branchpoints');

nexttile; 
imshow(BW_cleaned_parts); 
title(['3. Đã xóa đoạn < ' num2str(BRIDGE_LIMIT) 'px']);

nexttile; 
imshow(BW_final_skel); 
title('4. Kết quả (Đã hàn lại)');

% Hiển thị overlay lên ảnh gốc để xem có mất vân thật không
figure; imshowpair(hologram, BW_final_skel, 'montage');
title('So sánh: Ảnh gốc vs Skeleton cuối cùng');


%%
BW = skel_clean;
endPoints = bwmorph(BW, 'endpoints');

% --- Tính hướng vector tại endpoint ---
Nfit = 12;  % số pixel dùng để fit PCA
vectors = fitEndpointVectors(BW, endPoints, Nfit);

% --- Hiển thị kết quả ---
figure;
imshow(BW); hold on;
title('Hướng tại các Endpoint');
for i = 1:size(vectors, 1)
    cx = vectors(i,1);
    cy = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);

    % Vẽ vector hướng (mũi tên)
    quiver(cx, cy, vx*10, vy*10, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 2);
end
hold off;

%% noois diem
n = 4;
for count =1:n
    % Timf endPoint
    % Kernel để đếm số hàng xóm (8-neighbors)
    endPoints = bwmorph(BW, 'endpoints');
    figure; imshow(BW); title('Skeleton gốc');
    [row, col] = find(endPoints);
    hold on; plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    %
    fprintf('--> Bước 2b: Ước lượng vector hướng theo đoạn liên thông\n');
    vectors = fitEndpointVectors(BW, endPoints, 30);

    imshow(BW); hold on;
    for i = 1:size(vectors,1)
        cx = vectors(i,1);
        cy = vectors(i,2);
        vx = vectors(i,3);
        vy = vectors(i,4);
        quiver(cx, cy, 10*vx, 10*vy, 'r', 'LineWidth',2, 'MaxHeadSize',2);
    end
    title('Vector hướng tại các endpoint');
    
    %%
    fprintf('--> Bước 2c: Nối các endpoint theo vector hướng\n');
    if count == 1 %nối vân dài + góc lệch nhỏ + khoảng cách nhỏ
        fprintf('--> lan chay dau tien\n');
        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 6;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count ==2 % Nối vân dài + góc lệch lớn + khoảng cách lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 2\n');

        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 12;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 3 % Nối vân dài + góc lệch nhỏ + khoảng cách lớn hơn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 3\n');

        minCompSize = 12;   % chỉ nối nếu component đủ dài
        maxDist     = 25;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 4 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 4\n');

        minCompSize = 5;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(40);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 5 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 5\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 6 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 6\n');

        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 7 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 7\n');
        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 50;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(30);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 8 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 7\n');
        minCompSize = 20;   % chỉ nối nếu component đủ dài
        maxDist     = 60;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(45);  % = 0.866 ~ hướng lệch <= 30°
    end
end   
    CC = bwconncomp(BW, 8);

    [BW, linesConnected] = connectEndpoints(BW, vectors, CC, minCompSize, maxDist, vecAlignThr);
% xoá vùng nhỏ lẻ
    BW = removeSmallComponents(BW, 6);  % xoá vùng liên thông < 10 pixel

    figure; imshow(BW); hold on;
    for k = 1:numel(linesConnected)
        lineXY = linesConnected{k};
        plot(lineXY(:,1), lineXY(:,2), 'g-', 'LineWidth', 2);
    end
    title('Skeleton sau khi nối endpoint (màu xanh)');




    %%
    function vectors = fitEndpointVectors(BW, endPoints, Nfit)
% fitEndpointVectors - Tính vector hướng tại endpoint của skeleton
%
% Cú pháp:
%   vectors = fitEndpointVectors(BW, endPoints, Nfit)
%
% Input:
%   BW        - ảnh nhị phân skeleton
%   endPoints - ảnh nhị phân endpoint (1 tại endpoint)
%   Nfit      - số pixel dùng để fit PCA (ví dụ: 30)
%
% Output:
%   vectors - ma trận [N x 4], mỗi hàng:
%             [cx cy vx vy]
%             (cx, cy) = tọa độ endpoint
%             (vx, vy) = vector đơn vị hướng ra ngoài

    [y_idx, x_idx] = find(endPoints);  % tọa độ endpoints
    CC = bwconncomp(BW, 8);           % tìm các component
    vectors = [];

    for k = 1:length(x_idx)
        cx = x_idx(k); 
        cy = y_idx(k);

        % Kiểm tra endpoint thuộc component nào
        comp_id = 0;
        for c = 1:CC.NumObjects
            if ismember(sub2ind(size(BW), cy, cx), CC.PixelIdxList{c})
                comp_id = c; 
                break;
            end
        end

        if comp_id == 0, continue; end  % endpoint không thuộc component nào

        % Lấy tọa độ tất cả pixel trong component
        [yy, xx] = ind2sub(size(BW), CC.PixelIdxList{comp_id});

        % Tính khoảng cách từ endpoint
        dist2 = (xx - cx).^2 + (yy - cy).^2;
        [~, idx] = sort(dist2);
        idxN = idx(1:min(Nfit, numel(idx)));

        X = xx(idxN); 
        Y = yy(idxN);

        if numel(X) > 1
            % --- Fit hướng bằng PCA ---
            Xc = X - mean(X);
            Yc = Y - mean(Y);
            D = [Xc(:) Yc(:)];
            [~,~,V] = svd(D,'econ');
            v = V(:,1);  % vector chính (cột đầu tiên)
            v = v / norm(v);

            % --- Xác định hướng "ra ngoài" ---
            centroid = [mean(X); mean(Y)];
            c = centroid - [cx; cy];  % vector từ endpoint vào trong component
            if dot(v, c) > 0
                v = -v; % đảo dấu để hướng ra ngoài
            end
        else
            v = [0;0];
        end

        vectors = [vectors; cx cy v(1) v(2)];
    end
end
function [BW_new, linesConnected] = connectEndpoints(BW, vectors, CC, minCompSize, maxDist, vecAlignThr)
% BW           : skeleton binary
% vectors      : [cx cy vx vy] từ hàm computeEndpointVectors
% CC           : bwconncomp(BW,8)
% minCompSize  : kích thước tối thiểu của vân
% maxDist      : khoảng cách tối đa cho phép nối
% vecAlignThr  : ngưỡng cos(angle) hướng (ví dụ 0.7 ~ >45°)

BW_new = BW; % copy để cập nhật nối
linesConnected = {}; % cell lưu danh sách các đoạn đã nối

for i = 1:size(vectors,1)-1
    cx1 = vectors(i,1); cy1 = vectors(i,2);
    v1  = [vectors(i,3), vectors(i,4)];
    
    % kiểm tra component của endpoint i
    comp_id1 = findComponent(CC, [cy1,cx1]);
    if comp_id1==0 || numel(CC.PixelIdxList{comp_id1}) < minCompSize
        continue;
    end
    
    for j = i+1:size(vectors,1)
        cx2 = vectors(j,1); cy2 = vectors(j,2);
        v2  = [vectors(j,3), vectors(j,4)];

        % kiểm tra component j
        comp_id2 = findComponent(CC, [cy2,cx2]);
        if comp_id2==0 || numel(CC.PixelIdxList{comp_id2}) < minCompSize
            continue;
        end
        
        % --- khoảng cách Euclidean giữa 2 endpoint ---
        d = hypot(cx1-cx2, cy1-cy2);
        if d > maxDist, continue; end

        % --- kiểm tra hướng vector (cùng hướng nối) ---
        dir12 = [cx2-cx1, cy2-cy1];
        dir12 = dir12 / (norm(dir12)+eps);

        cond1 = dot(v1, dir12) > vecAlignThr;    % v1 hướng về P2
        cond2 = dot(v2, -dir12) > vecAlignThr;   % v2 hướng về P1

        if ~(cond1 && cond2), continue; end

        % --- kiểm tra thêm khoảng cách vuông góc ---
        % đường thẳng qua P2 với vector v2
        a = -v2(2);
        b =  v2(1);
        c =  v2(2)*cx2 - v2(1)*cy2;
        d_perp = abs(a*cx1 + b*cy1 + c) / sqrt(a^2 + b^2);

        if d_perp > 5, continue; end

        % --- nối 2 endpoint ---
        [BW_new, linePixels] = drawLine(BW_new, cx1, cy1, cx2, cy2);
        linesConnected{end+1} = linePixels; %#ok<AGROW>
    end
end
end
