clc; clear; close all;

% --- Skeleton giả định bị đứt ---
BW = false(100,100);
BW(20:40,30) = 1;          % đoạn 1
BW(41:43,31) = 1;          % đoạn ngắn lệch
BW(44:60,32) = 1;          % đoạn 2
BW(20:60,70) = 1;          % vân song song khác

imshow(BW); title('Skeleton bị đứt'); hold on;

% --- Connected components ---
CC = bwconncomp(BW);
stats = regionprops(CC,'PixelList','Centroid');

% Gom các cụm thành graph theo khoảng cách centroid
n = CC.NumObjects;
centroids = cat(1, stats.Centroid);

% Tính khoảng cách giữa các cụm
D = pdist2(centroids,centroids);

% Đặt ngưỡng nối (ví dụ 10 pixel)
thresh = 10;

G = graph(D<thresh & D>0);

% Tìm các nhóm (clusters) = các vân
comp = conncomp(G);

colors = lines(max(comp));
for k=1:max(comp)
    idx = find(comp==k);
    pts = [];
    for i=idx
        pts = [pts; stats(i).PixelList];
    end
    % Fit spline cho nhóm
    [~,ord] = sort(pts(:,1));
    pts = pts(ord,:);
    xx = linspace(min(pts(:,1)),max(pts(:,1)),200);
    yy = spline(pts(:,1),pts(:,2),xx);
    plot(xx,yy,'-','Color',colors(k,:),'LineWidth',2);
end
