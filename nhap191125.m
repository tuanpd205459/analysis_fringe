clc, clear, close all;
addpath("C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 12 11 25\251112-20251112T152447Z-1-001\251112");

% % --- 1. CHỌN 1 FILE ẢNH ĐỂ XỬ LÝ ---
% fprintf('Vui lòng chọn 1 file ảnh để xử lý...\n');
% [filename, folderPath] = uigetfile({'*.bmp'}, 'Chọn 1 file ảnh');
% 
% % Kiểm tra nếu người dùng nhấn 'Cancel'
% if isequal(filename, 0) || isequal(folderPath, 0)
%     fprintf('Bạn đã hủy. Dừng chương trình.\n');
%     return;
% end
% 
% % Ghép tên file và đường dẫn
% imgPath = fullfile(folderPath, filename);
% fprintf('Đang xử lý file: %s\n', imgPath);
imgPath=("C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 12 11 25\60x o thu 6\251112\anh oke\image_2025-11-12T18-33-12.732.bmp");
hologram = imread(imgPath);
if size(hologram, 3) == 3
    hologram = rgb2gray(hologram);
    fprintf('Đã chuyển đổi ảnh RGB sang grayscale\n');
end

hologram = rot90(hologram);   % Xoay 180 độ (mỗi lần rot90 xoay 90 độ)

%% 5. Noise removal

hologram = imgaussfilt(hologram, 1);

%% 6. Histogram equalization

hologram = adapthisteq(hologram);
figure;
imshow(hologram);
colorbar;
title('hologram sau noise adapthisteq : ');
%%
sensitive_coef = 0.65;
nei = 51;

T = adaptthresh(hologram, sensitive_coef, ...
    'NeighborhoodSize', [nei nei], ...    % kích thước vùng cục bộ
    'Statistic', 'median');           % hoặc 'mean'

% Nhị phân hóa ảnh
hologram_bin = imbinarize(hologram, T);

figure;
imshow(hologram_bin);
title('Adaptive threshold theo từng vùng median');

%%
figure;
imshow(hologram_bin);
title('Cắt ngưỡng ảnh tốt');



%% 
hologram_bin =bwmorph(hologram_bin,"thicken", 1);
hologram_bin = bwmorph(hologram_bin,"close");

% --- Bước 2: Skeletonize bằng Zhang-Suen ---
fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');

BW_Original = hologram_bin;


BW = bwmorph(hologram_bin, "skeleton", Inf) ;
figure;
imshowpair(BW, hologram_bin, 'falsecolor');   % Hiển thị 2 ảnh bằng 2 kênh màu khác nhau
title('Overlay bằng falsecolor');

%
BW = MZS_thinning(BW);


%% loại bỏ các điểm rời rạc có kích thước nhỏ:

BW = imfill(BW,"holes");
BW = bwmorph(BW, "skeleton", Inf);
BW = bwmorph(BW,"clean", Inf);
%%
%%
distThresh = 8;
skel = BW;
% --- Chuẩn hóa ảnh ---
skel = logical(skel);
skel_clean = skel;

% --- Tìm endpoint và branchpoint ---
endpoints = bwmorph(skel, 'endpoints');
branchpoints = bwmorph(skel, 'branchpoints');

[yE, xE] = find(endpoints);
[yB, xB] = find(branchpoints);

% --- Tìm các vùng liên thông trong skeleton ---
CC = bwconncomp(skel);

for i = 1:numel(xE)
    ptE = [yE(i), xE(i)];

    % Kiểm tra vùng chứa endpoint này
    compIdx = 0;
    for c = 1:CC.NumObjects
        if ismember(sub2ind(size(skel), ptE(1), ptE(2)), CC.PixelIdxList{c})
            compIdx = c;
            break;
        end
    end
    if compIdx == 0, continue; end % không thuộc vùng nào (hiếm gặp)

    % Lấy các branchpoints thuộc cùng vùng
    bpMask = false(size(skel));
    bpMask(CC.PixelIdxList{compIdx}) = true;
    [yB_comp, xB_comp] = find(branchpoints & bpMask);

    if isempty(xB_comp), continue; end

    % Tính khoảng cách đến branchpoints trong cùng vùng
    dists = sqrt((yB_comp - ptE(1)).^2 + (xB_comp - ptE(2)).^2);
    [minDist, idxMin] = min(dists);

    if minDist < distThresh
        ptB = [yB_comp(idxMin), xB_comp(idxMin)];

        % --- Tìm đường đi trong skeleton giữa 2 điểm ---
        D1 = bwdistgeodesic(skel, ptE(2), ptE(1));
        D2 = bwdistgeodesic(skel, ptB(2), ptB(1));

        if isempty(D1) || isempty(D2)
            continue; % trường hợp điểm bị ngắt kết nối
        end

        Dsum = D1 + D2;
        Dsum(~skel) = inf;

        pathMask = imregionalmin(Dsum);

        % --- Xóa đường nối đó ---
        skel_clean(pathMask) = 0;
    end
end

BW = skel_clean;
% BW = bwmorph(BW,"bridge",Inf);
% BW = bwmorph(BW,"diag", Inf);
% BW = bwmorph(BW,"skeleton", Inf);
% BW = bwmorph(BW,'spur',5);

figure;
imshow(BW); title('Skeleton sau khi xóa nhánh nhỏ');





%%
function S = MZS_thinning(BW)
    S = BW > 0;
    prev = false(size(S));
    while true
        % Sub-iteration 1: even pixels
        marker = MZS_iteration(S, 0);
        S(marker) = 0;

        % Sub-iteration 2: odd pixels
        marker = MZS_iteration(S, 1);
        S(marker) = 0;

        if isequal(S, prev), break; end
        prev = S;
    end
end
function marker = MZS_iteration(S, parity)
    % Pad để tránh lỗi biên
    P = padarray(S, [1 1], 0, 'both');

    % Lấy lân cận (theo thứ tự ZS)
    P2 = P(1:end-2,2:end-1); % north
    P3 = P(1:end-2,3:end);   % northeast
    P4 = P(2:end-1,3:end);   % east
    P5 = P(3:end,3:end);     % southeast
    P6 = P(3:end,2:end-1);   % south
    P7 = P(3:end,1:end-2);   % southwest
    P8 = P(2:end-1,1:end-2); % west
    P9 = P(1:end-2,1:end-2); % northwest
    P1 = P(2:end-1,2:end-1); % center

    % Tổng số hàng xóm (B)
    B = P2+P3+P4+P5+P6+P7+P8+P9;

    % C(p1) theo công thức trong paper
    C = (~P2 & (P3|P4)) + (~P4 & (P5|P6)) + ...
        (~P6 & (P7|P8)) + (~P8 & (P9|P2));

    % Điều kiện chung
    cond = (C==1);

    if parity == 0   % subfield chẵn
        cond = cond & (mod(bsxfun(@plus,(1:size(S,1))',(1:size(S,2))),2)==0);
        cond = cond & (B>=2 & B<=7);
        cond = cond & (~(P2 & P4 & P6));
        cond = cond & (~(P4 & P6 & P8));
    else             % subfield lẻ
        cond = cond & (mod(bsxfun(@plus,(1:size(S,1))',(1:size(S,2))),2)==1);
        cond = cond & (B>=1 & B<=7);
        cond = cond & (~(P2 & P4 & P8));
        cond = cond & (~(P2 & P6 & P8));

        % Bổ sung điều kiện giữ pixel để bảo toàn 2x2 / diagonal
        diagNeighbors = (P3|P5|P7|P9);
        cond = cond & ~( (B==1) & diagNeighbors );
    end

    marker = P1 & cond;
end
