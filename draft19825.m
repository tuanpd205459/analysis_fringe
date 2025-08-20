clc; clear; close all;

%% ===== 1) TẠO BỀ MẶT & SKELETON TỪ PEAKS =====
M = 200; 
[Xg, Yg, Ztrue] = peaks(M);
Ztrue = Ztrue - min(Ztrue(:));
lambda = 1; 
step = lambda/2;

% Contour mức cách nhau lambda/2 -> coi như tâm vân
levels = 0:step:max(Ztrue(:));
C = contourc(Ztrue, levels);

BW = false(M,M);
k = 1;
while k < size(C,2)
    lvl   = C(1,k);
    npt   = C(2,k);
    pts   = C(:,k+1:k+npt).';
    xy    = round(pts);                          % dùng pixel index
    xy(xy<1)=1; xy(xy>M)=M;
    BW(sub2ind([M M], xy(:,2), xy(:,1))) = true; % (row=y, col=x)
    k = k + npt + 1;
end
S = bwmorph(BW,'skel',Inf);
S = bwmorph(S,'spur',8);
S = bwareaopen(S,10);

%% ===== 2) GÁN NHÃN VÂN & GIÁ TRỊ λ/2 =====
cc = bwconncomp(S);
L = labelmatrix(cc);
num = max(L(:));
if num < 2, error('Cần >=2 đường vân.'); end

% Sắp thứ tự vân bằng PCA (ổn định với uốn cong)
stats = regionprops(L,'Centroid','PixelList');
P = vertcat(stats.PixelList);
P = double(P) - mean(double(P),1);
[U,~,~] = pca(P);
nperp = [-U(2,1); U(1,1)];
proj = arrayfun(@(s) dot((s.Centroid.'-mean(double(P)).'), nperp), stats);
[~,ord] = sort(proj,'ascend');
k_order = zeros(num,1);
for r=1:num, k_order(ord(r)) = r-1; end
z_line = k_order * step;                     % mỗi vân cách nhau λ/2

%% ===== 3) BFS VORONOI: VÂN GẦN NHẤT TOÀN ẢNH =====
% Dùng hàng đợi Java để BFS đa nguồn
[Mh, Nw] = size(S);
nearestLabel = zeros(Mh,Nw,'uint16');          % nhãn vân gần nhất
dist1 = inf(Mh,Nw);                            % khoảng cách tới vân gần nhất

Q = java.util.ArrayDeque();

% seed từ tất cả điểm skeleton
[yx, xx] = find(S);
for t=1:numel(xx)
    i = yx(t); j = xx(t);
    nearestLabel(i,j) = uint16(L(i,j));
    dist1(i,j) = 0;
    Q.add([i j]);
end

% 4-neighborhood
NBR = [0 1; 1 0; 0 -1; -1 0];

% BFS đa nguồn
while ~Q.isEmpty()
    p = Q.remove();
    i = p(1); j = p(2);
    for n=1:4
        ii = i + NBR(n,1);
        jj = j + NBR(n,2);
        if ii>=1 && ii<=Mh && jj>=1 && jj<=Nw
            dnew = dist1(i,j) + 1; % Manhattan; đủ tốt cho Voronoi rời rạc
            if dnew < dist1(ii,jj)
                dist1(ii,jj) = dnew;
                nearestLabel(ii,jj) = nearestLabel(i,j);
                Q.add([ii jj]);
            end
        end
    end
end

%% ===== 4) TÌM VÂN THỨ NHÌ + RAMP λ/2 (ổn định ở mép dải) =====
% Để lấy vân thứ nhì khác nhãn: tính lại EDT khi bỏ từng nhãn (đơn giản & rõ ràng)
SkelLabelImg = zeros(Mh,Nw,'uint16'); SkelLabelImg(S) = uint16(L(S));
[D1, idx1] = bwdist(S, 'euclidean');        % khoảng cách tới skeleton gần nhất (chuẩn hơn dist1)
lab1 = SkelLabelImg(idx1);                  % nhãn vân gần nhất theo EDT

Z_ramp = nan(Mh,Nw);
D2_all = inf(Mh,Nw); lab2_all = zeros(Mh,Nw,'uint16');

for a = 1:num
    Sa = S; Sa(L==a) = false;
    if ~any(Sa(:)), continue; end
    [D2, idx2] = bwdist(Sa,'euclidean');
    tmpLab = zeros(Mh,Nw,'uint16'); tmpLab(Sa) = uint16(L(Sa));
    lab2 = tmpLab(idx2);

    mask = (lab1==a) & (lab2>0);
    d1 = D1(mask); d2 = D2(mask);

    v1 = z_line(a) * ones(nnz(mask),1);
    v2 = z_line(double(lab2(mask)));           % giá trị vân thứ nhì

    % tham số t nội suy theo khoảng cách: đi từ v1 -> v2
    t = d1 ./ (d1 + d2 + eps);
    Z_ramp(mask) = (1 - t).*v1 + t.*v2;        % đảm bảo λ/2 giữa 2 vân
    % Lưu lại d2, lab2 cho pha trộn sau
    D2_all(mask) = d2;
    lab2_all(mask) = lab2(mask);
end

% Điểm không có vân thứ nhì (rìa ngoài): gán theo vân gần nhất
remain = isnan(Z_ramp);
if any(remain(:))
    Z_ramp(remain) = z_line(double(lab1(remain)));
end

%% ===== 5) SCATTEREDINTERPOLANT TOÀN CỤC (mượt) =====
[y_s, x_s] = find(S);
z_s = z_line(double(L(S)));     % giá trị z tại skeleton
F = scatteredInterpolant(double(x_s), double(y_s), double(z_s), 'natural', 'nearest');
[Xq, Yq] = meshgrid(1:Nw,1:Mh);
Z_scat = F(Xq, Yq);

%% ===== 6) PHA TRỘN THÔNG MINH: NEAR-EDGE ưu tiên RAMP, MID-BAND ưu tiên SCAT =====
% Thước đo "ở giữa dải": tau = |d1 - d2| / (d1 + d2)
hasSecond = lab2_all > 0;
tau = ones(Mh,Nw);                                  % biên mặc định 1 (ưu tiên ramp)
tau(hasSecond) = abs(D1(hasSecond) - D2_all(hasSecond)) ./ ...
                 (D1(hasSecond) + D2_all(hasSecond) + eps);
% Ở giữa dải: tau ~ 0; ở sát vân: tau ~ 1
w = 1 - tau;                                        % trọng số cho SCAT (giữa dải cao hơn)
w = max(0,min(1,w));
% Tuỳ chọn làm mượt trọng số một chút để không gắt:
w = imgaussfilt(w, 1);

Z_hybrid = (1 - w).*Z_ramp + w.*Z_scat;

%% ===== 7) HẬU XỬ LÝ NHẸ (bỏ tilt & offset, optional) =====
Z_out = Z_hybrid;
% Bỏ offset
Z_out = Z_out - median(Z_out(:));

%% ===== 8) SO SÁNH VỚI TRUE SURFACE (đã scale theo λ/2) =====
% Đưa Ztrue về cùng "nấc" với hệ λ/2 (chỉ để trực quan; không bắt buộc)
Zt = Ztrue; Zt = Zt - median(Zt(:));
% Chuẩn hóa biên độ để so sánh tương đối (không thay đổi bản chất demo)
scale = median(abs(Z_out(:))) / (median(abs(Zt(:))) + eps);
Zt_scaled = Zt * scale;

err_scat = Z_scat - Zt_scaled;
err_ramp = Z_ramp - Zt_scaled;
err_hyb  = Z_out  - Zt_scaled;

%% ===== 9) HIỂN THỊ =====
figure('Name','Hybrid BFS + ScatteredInterpolant'); 
subplot(2,3,1); imagesc(Zt_scaled); axis image off; colorbar; title('True (scaled)');
subplot(2,3,2); imshow(S); title('Skeleton');
subplot(2,3,3); imagesc(Z_scat); axis image off; colorbar; title('ScatteredInterpolant');

subplot(2,3,4); imagesc(Z_ramp); axis image off; colorbar; title('Ramp λ/2 (BFS/EDT-based)');
subplot(2,3,5); imagesc(Z_out);  axis image off; colorbar; title('Hybrid = blend(ramp, scat)');
subplot(2,3,6); imagesc(err_hyb); axis image off; colorbar; title(sprintf('Error Hybrid (RMSE=%.4f)', rms(err_hyb(:))));

figure('Name','Errors');
subplot(1,3,1); imagesc(err_scat); axis image off; colorbar; title(sprintf('Scattered err (RMSE=%.4f)', rms(err_scat(:))));
subplot(1,3,2); imagesc(err_ramp); axis image off; colorbar; title(sprintf('Ramp err (RMSE=%.4f)',  rms(err_ramp(:))));
subplot(1,3,3); imagesc(err_hyb);  axis image off; colorbar; title(sprintf('Hybrid err (RMSE=%.4f)', rms(err_hyb(:))));
