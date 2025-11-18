clc; clear; close all;

%% load ảnh thực

addpath("C:\Users\admin\Máy tính\Lab thầy Tùng\My_TN\data 13 11 25\guong 60x");
hologram = imread("image_2025-11-13T17-50-34.034.bmp");
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
sensitive_coef = 0.6;
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

% --- 3. Gọi hàm myDrawRec() ---
[pos, xRec, yRec, widthRec, heightRec] = myDrawRec();

% --- 4. Cắt vùng ảnh đã chọn ---
hologram_bin_crop = hologram(yRec : yRec + heightRec - 1, ...
                         xRec : xRec + widthRec - 1);
hologram_bin = hologram_bin_crop;

figure();
imshow(hologram_bin, []);
title('Sau khi cat');
%%
sensitive_coef = 0.6;
nei = 51;

T = adaptthresh(hologram_bin, sensitive_coef, ...
    'NeighborhoodSize', [nei nei], ...    % kích thước vùng cục bộ
    'Statistic', 'median');           % hoặc 'mean'

% Nhị phân hóa ảnh
hologram_bin = imbinarize(hologram_bin, T);

figure;
imshow(hologram_bin);
title('Adaptive threshold sau khi cắt');

%%
% --- Bước 2: Skeletonize bằng Zhang-Suen ---
    fprintf('Bước 2/3: Áp dụng thuật toán Zhang-Suen...\n');
BW_Original = hologram_bin;
BW_Thinned = BW_Original;
[rows, cols] = size(BW_Thinned);
changing = true;
iteration = 0;

while changing
    iteration = iteration + 1;
    changing = false;
    BW_Del = true(rows, cols);

    % --- Step 1 của Zhang-Suen ---
    for i = 2:rows-1
        for j = 2:cols-1
            P = BW_Thinned(i-1:i+1, j-1:j+1);
            P = P(:)';
            % Sắp xếp theo thứ tự: P1(center), P2, P3, P4, P5, P6, P7, P8, P9, P2(lặp)
            P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

            if P(1) == 1  % Nếu pixel trung tâm là foreground
                neighbors = sum(P(2:9));  % Số lượng neighbor foreground
                transitions = sum(P(2:9) == 0 & P(3:10) == 1);  % Số transition 0->1

                % Điều kiện Zhang-Suen Step 1
                if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                        && P(2)*P(4)*P(6) == 0 && P(4)*P(6)*P(8) == 0
                    BW_Del(i,j) = false;
                    changing = true;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned & BW_Del;

    % --- Step 2 của Zhang-Suen ---
    BW_Del = true(rows, cols);
    for i = 2:rows-1
        for j = 2:cols-1
            P = BW_Thinned(i-1:i+1, j-1:j+1);
            P = P(:)';
            P = [P(5), P(2), P(3), P(6), P(9), P(8), P(7), P(4), P(1), P(2)];

            if P(1) == 1
                neighbors = sum(P(2:9));
                transitions = sum(P(2:9) == 0 & P(3:10) == 1);

                % Điều kiện Zhang-Suen Step 2
                if neighbors >= 2 && neighbors <= 6 && transitions == 1 ...
                        && P(2)*P(4)*P(8) == 0 && P(2)*P(6)*P(8) == 0
                    BW_Del(i,j) = false;
                    changing = true;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned & BW_Del;

    % Hiển thị tiến trình mỗi 10 iterations
    if mod(iteration, 10) == 0
        %             fprintf('  Iteration %d: %d pixels còn lại\n', iteration, sum(BW_Thinned(:)));
    end

    % Tránh vòng lặp vô hạn
    if iteration > 1000
        warning('Đã đạt giới hạn iteration (1000). Dừng thuật toán.');
        break;
    end
end
%%
% --- Trả về kết quả ---
skeleton_image = BW_Thinned;
binary_image = BW_Original;

skeleton = skeleton_image;
BW = skeleton;

%
S = MZS_thinning(BW);
BW = S;

figure;
imshowpair(BW, hologram_bin, 'falsecolor');   % Hiển thị 2 ảnh bằng 2 kênh màu khác nhau
title('Overlay bằng falsecolor');

%% loại bỏ các điểm rời rạc có kích thước nhỏ:
BW = imfill(BW,"holes");
BW = bwmorph(BW, "skeleton", Inf);
BW = bwmorph(BW,"clean", Inf);
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

% --- Hiển thị kết quả ---
figure;
subplot(1,2,1);
imshow(skel); title('Skeleton gốc');

subplot(1,2,2);
imshow(skel_clean); title('Skeleton sau khi xóa nhánh nhỏ');


%% --- Tìm endpoint ---
BW = skel_clean;
BW = bwmorph(BW,"bridge",Inf);
BW = bwmorph(BW,"diag", Inf);
BW = bwmorph(BW,"skeleton", Inf);
BW = bwmorph(BW,'spur',5);

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
% CC = bwconncomp(BW);
% minCompSize = 12;   % chỉ nối nếu component đủ dài
% maxDist     = 40;   % khoảng cách tối đa giữa 2 endpoint
% vecAlignThr = cosd(50);  % = 0.866 ~ hướng lệch <= 30°
% BW = BW(2:end-1, 2:end-1);
% [BW_new, linesConnected] = connectEndpoints(BW, vectors, CC, minCompSize, maxDist, vecAlignThr);
% 
% 
% 
% lineMask = false(size(BW_new));
% 
% for k = 1:numel(linesConnected)
%     xy = linesConnected{k};
%     if isempty(xy), continue; end
% 
%     x = round(xy(:,1));
%     y = round(xy(:,2));
%     x = max(1, min(size(BW_new,2), x));
%     y = max(1, min(size(BW_new,1), y));
%     xy_valid = unique([x y], 'rows');
% 
%     idx = sub2ind(size(BW_new), xy_valid(:,2), xy_valid(:,1)); % (y,x)
%     lineMask(idx) = true;
% end
% 
% figure;
% imshow(BW_new); hold on;
% title('Các đoạn đã nối (màu đỏ)');
% redOverlay = cat(3, ones(size(BW_new)), zeros(size(BW_new)), zeros(size(BW_new)));
% h = imshow(redOverlay);
% set(h, 'AlphaData', lineMask * 0.7);
% hold off;

n = 1;
for count =1:n
    % Timf endPoint
    % Kernel để đếm số hàng xóm (8-neighbors)
    endPoints = bwmorph(BW, 'endpoints');
    figure; imshow(BW); title('Skeleton gốc');
    [row, col] = find(endPoints);
    hold on; plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    %
    fprintf('--> Bước 2b: Ước lượng vector hướng theo đoạn liên thông\n');
    vectors = fitEndpointVectors(BW, endPoints, 12);

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
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
    end
    if count == 4 % Nối vân ngắn + góc lệch lớn
        fprintf('--> Bước 2d: Nối các endpoint theo vector hướng\n');
        fprintf('--> lan chay lan 4\n');

        minCompSize = 5;   % chỉ nối nếu component đủ dài
        maxDist     = 20;   % khoảng cách tối đa giữa 2 endpoint
        vecAlignThr = cosd(15);  % = 0.866 ~ hướng lệch <= 30°
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

end
%%
%% 7. ƯỚC LƯỢNG PHA BẰNG PHƯƠNG PHÁP PHÂN TÍCH VÂN
fprintf('--> Bước 3: Ước lượng pha thô bằng phân tích vân...\n');
% Làm mảnh và gán bậc vân

[~, labels, img] = assign_fringe_order(skeleton_image, true);

% Tái tạo bề mặt từ vân
[phi_est, ~] = reconSurface_linearPushed(img, labels, 632.8e-9, 'Remove tilt', false);

% Sau khi có phi_est
% systematic_error = (pi/5) * (2*rand(size(phi_est))-1); % random ±λ/20
% phi_est = phi_est + systematic_error;

phi_est = phi_est - min(phi_est(:));


figure;
surf(phi_est,"EdgeColor","none");
title("Anh pha phi estimate co nhieu");


%% 8. GIẢI BỌC PHA VÀ TINH CHỈNH
fprintf('--> Bước 4: Giải bọc pha và tinh chỉnh kết quả...\n');
% --- Giải bọc pha sử dụng pha ước lượng ---
% [est_phase_flat, wrapped_phase, object_phase] = crop_multiple_to_smallest(est_phase_flat, wrapped_phase, object_phase);

[finalUnwrappedPhase, kMap] = unwrapUsingEstimate(phi_est, wrapped_phase);


%% 10. Refine artifacts points

% Cắt biên để hiển thị tốt hơn
offset = 10;
finalUnwrappedPhase = finalUnwrappedPhase(offset+1:end-offset, offset+1:end-offset);

%% 11. CÁC THUẬT TOÁN UNWRAPPING KHÁC
unwrapped_Phase_LS_DCT = unwrapping.unwrapPhase(wrapped_phase, 'ls', 'dct'); % LS với DCT
unwrapped_Phase_TIE_FFT = unwrapping.unwrapPhase(wrapped_phase, 'tie', 'fft'); % TIE với FFT
unwrapped_Phase_noncontinue = unwrapping.unwrapPhase(wrapped_phase, 'linh'); % Phương pháp của a Linh
unwrapped_Phase_2dweight = unwrapping.unwrapPhase(wrapped_phase, '2dweight'); % 2D weighted phase unwrapping
unwrapped_Phase_goldstein = goldstein_unwrap(wrapped_phase);
% proposal 
unwrapped_Phase_proposal = finalUnwrappedPhase;
[object_phase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
    unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, unwrapped_Phase_proposal]...
    = crop_multiple_to_smallest(object_phase, unwrapped_Phase_LS_DCT, unwrapped_Phase_TIE_FFT, unwrapped_Phase_noncontinue,...
    unwrapped_Phase_2dweight, unwrapped_Phase_goldstein, unwrapped_Phase_proposal);
[M,N] = size(object_phase);

%% 6. PHÂN TÍCH SAI SỐ (TIẾP THEO)
% close all;

% --- Các thuật toán cần so sánh ---
   % --- B2: gom lại các bề mặt pha cần đánh giá ---
    phases = { unwrapped_Phase_TIE_FFT, ...
               unwrapped_Phase_noncontinue, unwrapped_Phase_2dweight, ...
               unwrapped_Phase_goldstein, unwrapped_Phase_proposal };

    % --- B3: tính sai số cho từng thuật toán ---
    for k = 1:nPhase
        phase = phases{k};

        % Loại offset
        offset = median(phase(:), 'omitnan') - median(object_phase(:), 'omitnan');
        phase_adj = phase - offset;

        % Sai số (scalar)
        diff = phase_adj - object_phase;
        rmse = sqrt(mean(diff(:).^2, 'omitnan'));

        % Lưu đúng 1 số
        results_rmse_all(k,idxSNR) = rmse;
    end

%% HIỂN THỊ KẾT QUẢ SAU TOÀN BỘ VÒNG LẶP
fprintf('\n==== TỔNG HỢP KẾT QUẢ RMSE ====\n');
for k = nPhase:-1:1
    fprintf('\n%s:\n', phase_names{k});
    for idxSNR = 1:nSNR
        fprintf('  Nhiễu = %2d %:  RMSE = %.4e\n', noise_percent(idxSNR), results_rmse_all(k,idxSNR));
    end
end

%% VẼ ĐỒ THỊ
figure;
hold on; grid on; box on;

% Định nghĩa marker cho từng thuật toán
markers = {'x','s','d','^','o'};
% Định nghĩa line style khác nhau
lineStyles = {'-','--','-.',':','-'};

lineWidth_size = {1.5,1.5,1.5,1.5,1.5};
for k = nPhase:-1:1
    plot(noise_percent, results_rmse_all(k,:), ...
        '-','LineWidth',lineWidth_size{k}, ...
        'Marker', markers{k}, ...
        'DisplayName', phase_names{k});
end

xlabel('% nhiễu');
ylabel('RMSE');
title('So sánh RMSE của các thuật toán theo % nhiễu');
legend('Location','eastoutside'); 
set(gca,'YScale','log'); % log scale để phân biệt rõ
set(gca,'FontSize',12);  





%% cac ham phụ trợ
function [pos, xRec, yRec, widthRec, heightRec] = myDrawRec()
% Cho phép người dùng vẽ một hình chữ nhật (ROI) trên ảnh hiện tại.
hFig = gcf;
hROI = drawrectangle();
centerRec = [hROI.Position(1) + hROI.Position(3)/2, hROI.Position(2) + hROI.Position(4)/2];
hold on;
hMarker = plot(centerRec(1), centerRec(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
addlistener(hROI, 'MovingROI', @(src, evt) updateCenterRectangle(src, hMarker));

% Đợi người dùng double-click để xác nhận
wait(hROI);

pos = round(hROI.Position);
xRec = pos(1); yRec = pos(2);
widthRec = pos(3); heightRec = pos(4);

% Đóng cửa sổ sau khi đã chọn xong
if ishandle(hFig)
    close(hFig);
end
end
% -------------------------------------------------------------------------
function updateCenterRectangle(roi, centerMarker)
% Cập nhật vị trí dấu cộng ở tâm ROI khi đang di chuyển.
centerMarker.XData = roi.Position(1) + roi.Position(3)/2;
centerMarker.YData = roi.Position(2) + roi.Position(4)/2;
drawnow;
end
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
function BW_clean = removeSmallComponents(BW, minSize)
% Xoá các vùng liên thông có kích thước < minSize pixel
%
% Input:
%   BW      - ảnh nhị phân (0/1)
%   minSize - ngưỡng số pixel nhỏ nhất giữ lại
%
% Output:
%   BW_clean - ảnh nhị phân sau khi lọc

% Tìm vùng liên thông (8-neighbors)
CC = bwconncomp(BW, 8);
% Đếm số pixel trong từng vùng
numPixels = cellfun(@numel, CC.PixelIdxList);

% Giữ lại vùng đủ lớn
BW_clean = BW;
for i = 1:CC.NumObjects
    if numPixels(i) < minSize
        BW_clean(CC.PixelIdxList{i}) = 0; % xoá vùng nhỏ
    end
end
%     BW_clean = bwmorph(BW_clean, 'spur', 1);  % loại bỏ các nhánh nhỏ lẻ

end
function [BW_clean, junction] = removeJunctions(BW)
% REMOVEJUNCTIONS - Phát hiện và loại bỏ junction pixels trong skeleton
%
% Syntax:
%   [BW_clean, junction] = removeJunctions(BW)
%
% Input:
%   BW - ảnh nhị phân skeleton
%
% Output:
%   BW_clean - skeleton sau khi xoá junction
%   junction - ảnh nhị phân, 1 tại vị trí junction pixel
%
% Đặc điểm:
%   - Junction pixel: có >=3 hàng xóm trong 8 hướng
%   - Bỏ qua các pixel sát biên (4 pixel)

% --- 1. Đếm số hàng xóm ---
kernel = ones(3,3); kernel(2,2) = 0;  % 8-neighborhood
neighborCount = conv2(double(BW), kernel, 'same');

% --- 2. Junction: skeleton pixel có >= 3 hàng xóm ---
junction = (BW == 1) & (neighborCount >= 3);

% --- 3. Không xét biên (4 pixel) ---
junction(1,:)       = 0;
junction(end,:) = 0;
junction(:,1)       = 0;
junction(:,end) = 0;

% --- 4. Xóa junction ---
BW_clean = BW;
BW_clean(junction) = 0;
end

function [BW, linePix] = drawLine(BW, x1, y1, x2, y2)
% Vẽ line nối từ (x1,y1) đến (x2,y2) bằng thuật toán Bresenham
[h, w] = size(BW);
[lineX, lineY] = bresenham(x1, y1, x2, y2);

linePix = [lineX(:), lineY(:)];

for k = 1:length(lineX)
    cx = lineX(k);
    cy = lineY(k);
    if cx >= 1 && cx <= w && cy >= 1 && cy <= h
        BW(cy, cx) = 1;
    end
end

end

function [x, y] = bresenham(x1, y1, x2, y2)

% Thuật toán Bresenham
x1 = round(x1); y1 = round(y1);
x2 = round(x2); y2 = round(y2);

dx = abs(x2 - x1);
dy = abs(y2 - y1);

sx = sign(x2 - x1);
sy = sign(y2 - y1);

err = dx - dy;

x = []; y = [];
while true
    x(end+1) = x1;
    y(end+1) = y1;
    if x1 == x2 && y1 == y2
        break;
    end
    e2 = 2 * err;
    if e2 > -dy
        err = err - dy;
        x1 = x1 + sx;
    end
    if e2 < dx
        err = err + dx;
        y1 = y1 + sy;
    end
end
end

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

function comp_id = findComponent(CC, p)
% p = [row, col]
comp_id = 0;
idx = sub2ind(CC.ImageSize, p(1), p(2));
for c = 1:CC.NumObjects
    if ismember(idx, CC.PixelIdxList{c})
        comp_id = c;
        return;
    end
end
end

function [unwrappedPhase, kMap] = unwrapUsingEstimate(estimatedPhase, wrappedPhase)
    % Giải Wrapped pha `wrappedPhase` dựa trên pha ước lượng `estimatedPhase`.
%     wrappedEstimate = wrapToPi(estimatedPhase);
    kMap = round((estimatedPhase - wrappedPhase) / (2*pi));
    unwrappedPhase = wrappedPhase + 2*pi * kMap;
end
function [fringe_order, fringe_labels, processed_image] = assign_fringe_order(input_image, display_result)
% ASSIGN_FRINGE_ORDER Gán bậc vân cho ảnh hologram đã được skeletonize
%
% Hàm này thực hiện gán nhãn bậc vân dựa trên vị trí tương đối so với tâm ảnh.
% Vân gần tâm nhất được gán bậc 0, các vân phía trên có bậc dương tăng dần,
% các vân phía dưới có bậc âm giảm dần.
%
% INPUT:
%   input_image    - Ảnh binary đã được skeletonize
%   display_result - (Optional) true/false để hiển thị kết quả (default: true)
%
% OUTPUT:
%   fringe_order     - Số lượng vân được phát hiện
%   fringe_labels    - Vector chứa nhãn bậc vân của từng vùng liên thông
%   processed_image  - Ảnh đã được cắt biên và xử lý
%
% EXAMPLE:
%   [order, labels, img] = assign_fringe_order(skeleton_image);
%   [order, labels, img] = assign_fringe_order(skeleton_image, false); % Không hiển thị

% --- Xử lý tham số đầu vào ---
if nargin < 1
    error('Thiếu tham số đầu vào: input_image');
end

if nargin < 2
    display_result = true; % Mặc định hiển thị kết quả
end

% --- Kiểm tra đầu vào ---
if isempty(input_image)
    error('Ảnh đầu vào không được để trống');
end

if ~islogical(input_image) && ~(isnumeric(input_image) && all(input_image(:) == 0 | input_image(:) == 1))
    error('Ảnh đầu vào phải là ảnh binary (logical hoặc 0/1)');
end

% Chuyển đổi sang logical nếu cần
if ~islogical(input_image)
    input_image = logical(input_image);
end

try
    % --- Bước 1: Cắt biên ảnh để tránh ảnh hưởng vùng biên ---
    offset = 0;
    [orig_H, orig_W] = size(input_image);

    % Kiểm tra kích thước ảnh
    if orig_H <= 2*offset || orig_W <= 2*offset
        warning('Ảnh quá nhỏ để cắt biên. Sử dụng ảnh gốc.');
        bw_crop = input_image;
        offset = 0;
    else
        bw_crop = input_image(offset+1:end-offset, offset+1:end-offset);
    end

    [H, W] = size(bw_crop);

    % --- Bước 2: Tìm các vùng liên thông (vân) ---

    cc = bwconncomp(bw_crop);

    if cc.NumObjects == 0
        warning('Không tìm thấy vân nào trong ảnh');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_crop;
        return;
    end

    labeled_matrix = labelmatrix(cc);
    stats = regionprops(cc, 'Centroid', 'BoundingBox');

    % --- Bước 3: Tìm nhóm gần tâm nhất làm gốc ---
    centroids = cat(1, stats.Centroid);
    image_center = [W/2, H/2];
    dist = vecnorm(centroids - image_center, 2, 2);
    [~, idx_center] = min(dist);

    % --- Bước 4: Khởi tạo và gán nhãn ---
    labels = nan(cc.NumObjects, 1);
    labels(idx_center) = 0; % Nhóm gốc đặt nhãn 0

    queue = idx_center; % Hàng đợi để duyệt lan truyền nhãn
    processed_groups = false(cc.NumObjects, 1);
    processed_groups(idx_center) = true;

    % --- Bước 5: Lan truyền nhãn ---
    while ~isempty(queue)
        current_group = queue(1);
        queue(1) = [];

        current_label = labels(current_group);
        pixels = cc.PixelIdxList{current_group};
        [rows, cols] = ind2sub([H, W], pixels);

        visited_gid = []; % Tránh xét lại nhóm cùng vòng lặp

        for i = 1:length(rows)
            r = rows(i);
            c = cols(i);

            % Lan truyền lên trên theo cột
            for y = r-1:-1:1
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label + 1; % Nhãn tăng dần lên trên
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end

            % Lan truyền xuống dưới theo cột
            for y = r+1:H
                gid = labeled_matrix(y, c);
                if gid > 0 && ~processed_groups(gid) && ~ismember(gid, visited_gid)
                    labels(gid) = current_label - 1; % Nhãn giảm dần xuống dưới
                    queue(end+1) = gid;
                    processed_groups(gid) = true;
                    visited_gid(end+1) = gid;
                    break;
                elseif gid > 0 && processed_groups(gid)
                    break;
                end
            end
        end
    end

    % --- Bước 6: Chuẩn hóa nhãn thành số nguyên dương bắt đầu từ 1 ---
    valid_labels = labels(~isnan(labels));

    if isempty(valid_labels)
        warning('Không thể gán nhãn cho bất kỳ vân nào');
        fringe_order = 0;
        fringe_labels = [];
        processed_image = bw_crop;
        return;
    end

    unique_labels = unique(valid_labels);
    labels_new = nan(size(labels));
    for i = 1:length(unique_labels)
        labels_new(labels == unique_labels(i)) = i;
    end
    labels = labels_new;

    % --- Bước 7: Hiển thị kết quả (nếu được yêu cầu) ---
    if display_result
        figure('Name', 'Kết quả gán bậc vân', 'NumberTitle', 'off');
        imshow(bw_crop);
        hold on;

        for k = 1:cc.NumObjects
            if ~isnan(labels(k))
                pixels = cc.PixelIdxList{k};
                [rows, cols] = ind2sub([H, W], pixels);
                coords = [cols, rows]; % [x, y]

                % Tính khoảng cách từ tâm ảnh để đặt nhãn ở vị trí gần tâm nhất
                dists = sqrt((coords(:,1) - image_center(1)).^2 + (coords(:,2) - image_center(2)).^2);
                [~, min_idx] = min(dists);
                label_pos = coords(min_idx, :);

                text(label_pos(1), label_pos(2), num2str(labels(k)), ...
                    'Color', 'r', 'FontSize', 11, 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center');
            end
        end

        title('Gán bậc vân', 'FontSize', 12);
        hold off;
    end

    % --- Bước 8: Trả về kết quả ---
    fringe_order = cc.NumObjects;
    fringe_labels = labels;
    processed_image = bw_crop;

%     % Hiển thị thống kê
%     fprintf('Đã phát hiện %d vân\n', fringe_order);
%     fprintf('Số vân được gán nhãn: %d\n', sum(~isnan(labels)));
%     if ~isempty(valid_labels)
%         fprintf('Phạm vi bậc vân: %d đến %d\n', min(unique_labels), max(unique_labels));
%     end

catch ME
    % Xử lý lỗi
    error_msg = sprintf('Lỗi trong quá trình gán bậc vân:\n%s', ME.message);
    error(error_msg);
end

end
function [recons_surface, figure_handle] = reconSurface_linearPushed(BW, fringe_labels, lambda, tilt_option, show_figure)
% RECONSURFACE_LINEARPUSHED Tái tạo bề mặt 3D từ ảnh vân giao thoa
%
% Cú pháp:
%   [recons_surface, figure_handle] = reconSurface_linearPushed(BW, fringe_labels, lambda, tilt_option, show_figure)
%
% Tham số đầu vào:
%   BW            - Ảnh nhị phân đã cắt biên (logical matrix)
%   fringe_labels - Vector chứa nhãn của các vân (double array)
%   lambda        - Bước sóng ánh sáng (double)
%   tilt_option   - Tùy chọn xử lý ('None', 'Remove tilt', 'Invert', 'Remove + Invert')
%   show_figure   - Có hiển thị figure hay không (logical, optional, default: true)
%
% Tham số đầu ra:
%   recons_surface - Ma trận bề mặt 3D đã tái tạo
%   figure_handle  - Handle của figure (nếu show_figure = true)
%
% Ví dụ:
%   [surface, fig] = reconSurface_linearPushed(BW, [1,2,3,4,5], 632.8e-9, 'Remove tilt');

% Xử lý tham số đầu vào
if nargin < 5
    show_figure = true;
end

% Kiểm tra tham số đầu vào
if isempty(fringe_labels)
    error('Bạn cần gán nhãn vân trước khi nội suy.');
end

if ~islogical(BW)
    error('BW phải là ảnh nhị phân (logical matrix).');
end

% Thiết lập khoảng cách giữa các vân
khoang_cach_van = lambda/2;

% Tìm các thành phần liên thông
cc = bwconncomp(BW);
L = labelmatrix(cc);

% Khởi tạo các mảng điểm 3D
num_labels = max(L(:));
X = []; Y = []; Z = [];

for i = 1:num_labels
    % Tìm các điểm thuộc vân có nhãn i
    [y, x] = find(L == i);

    if i <= length(fringe_labels)
        % Tính độ cao z dựa trên nhãn vân
        z = ones(size(x)) * (fringe_labels(i)) * khoang_cach_van;
        X = [X; x];
        Y = [Y; y];
        Z = [Z; z];
    end
end

% Kiểm tra xem có dữ liệu để nội suy không
if isempty(X)
    error('Không có dữ liệu để nội suy. Kiểm tra lại fringe_labels và BW.');
end

% Nội suy để tạo mặt 3D mượt
[xq, yq] = meshgrid(1:size(BW,2), 1:size(BW,1));
F = scatteredInterpolant(X, Y, Z, 'natural', 'nearest');
Zq = F(xq, yq);
Zq(~isfinite(Zq)) = 0;

% %
% Z_grid_cubic = griddata(X, Y, Z, xq, yq, 'cubic');
% Z_grid_cubic(~isfinite(Z_grid_cubic)) = 0;
% 
% % 6. Làm mượt hậu xử lý cho cubic
% Z_cubic_smooth = imgaussfilt(Z_grid_cubic, 2);
% Zq = Z_cubic_smooth;
% %

% Chuyển từ mét sang radian
phi_rad = (4 * pi / lambda) * Zq;
Zq = phi_rad;

% Cắt biên để hiển thị tốt hơn

Z_crop = Zq;


[M, N] = size(Z_crop);
[xGrid, yGrid] = meshgrid(1:N, 1:M);
x = xGrid(:);
y = yGrid(:);
z = Z_crop(:);

% Xử lý theo lựa chọn của người dùng
switch tilt_option
    case 'None'
        Z_processed = Z_crop;

    case 'Remove tilt'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_processed = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_processed = Z_crop - Z_fit;
        end

    case 'Invert'
        Z_processed = max(Z_crop(:)) - Z_crop;

    case 'Remove + Invert'
        good = ~isnan(z);
        if sum(good) < 3
            warning('Không đủ điểm hợp lệ để loại bỏ độ nghiêng.');
            Z_leveled = Z_crop;
        else
            A = [x, y, ones(size(x))];
            coeff = A(good,:) \ z(good);
            Z_fit = reshape(A * coeff, size(Z_crop));
            Z_leveled = Z_crop - Z_fit;
        end
        Z_processed = max(Z_leveled(:)) - Z_leveled;

    otherwise
        warning('Tùy chọn không hợp lệ. Sử dụng "None".');
        Z_processed = Z_crop;
end

% Chuẩn hóa bắt đầu từ 0
Z_offset = Z_processed - min(Z_processed(:));

% Gán kết quả đầu ra
recons_surface = Z_offset;

% Hiển thị bề mặt 3D nếu được yêu cầu
if show_figure
    figure_handle = figure;
    surf(xGrid, yGrid, Z_offset);
    shading interp;
    xlabel('X (px)');
    ylabel('Y (px)');
    zlabel('rad');
    title(['3D Surface Linear (Option: ', tilt_option, ')']);
    colormap parula;
    colorbar;
else
    figure_handle = [];
end

end
