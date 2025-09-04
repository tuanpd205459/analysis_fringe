clc; clear; close all;
BW = zeros(100,100);

% 2 điểm A và B
x1=20; y1=30;
x2=60; y2=50;

extendLen = 30; % nối dài thêm 30 pixel ra ngoài

BW2 = extendLineBeyondPoints(BW,x1,y1,x2,y2,extendLen);

imshow(BW2,[])
hold on;
plot([x1 x2],[y1 y2],'ro-') % A và B
function BW_out = extendLineBeyondPoints(BW, vectors, extendLen)
% extendLineBeyondPoints - Nối dài đoạn thẳng AB thêm 1 đoạn ở phía ngoài
%
% Input:
%   BW        - ảnh nhị phân
%   (x1,y1)   - điểm A
%   (x2,y2)   - điểm B
%   extendLen - số pixel muốn nối dài thêm
%            (cx, cy) = tọa độ endpoint
%             (vx, vy) = vector đơn vị hướng ra ngoài
% Output:
%   BW_out    - ảnh nhị phân sau khi vẽ đoạn thẳng nối dài

[H,W] = size(BW);
BW_out = BW;

% --- Vẽ endpoint và vector hướng ---
for i = 1:size(vectors,1)
    x2 = vectors(i,1);
    y2 = vectors(i,2);
    vx = vectors(i,3);
    vy = vectors(i,4);
end    

% điểm mới C = B + extendLen * v
x3 = x2 + extendLen*vx;
y3 = y2 + extendLen*vy;

% vẽ đường từ A đến C
[xLine, yLine] = bresenham(x1,y1, round(x3), round(y3));

% giữ pixel trong biên
mask = xLine>=1 & xLine<=W & yLine>=1 & yLine<=H;
xLine = xLine(mask);
yLine = yLine(mask);

% đánh dấu vào ảnh
BW_out(sub2ind([H,W], yLine, xLine)) = 1;

end

%% --- Hàm Bresenham ---
function [x,y] = bresenham(x1,y1,x2,y2)
x1=round(x1); y1=round(y1);
x2=round(x2); y2=round(y2);

dx=abs(x2-x1); dy=abs(y2-y1);
sx=sign(x2-x1); sy=sign(y2-y1);

x=x1; y=y1;
xx=[]; yy=[];

if dx > dy
    err = dx/2;
    while x ~= x2
        xx(end+1)=x; yy(end+1)=y;
        x = x + sx;
        err = err - dy;
        if err < 0
            y = y + sy;
            err = err + dx;
        end
    end
else
    err = dy/2;
    while y ~= y2
        xx(end+1)=x; yy(end+1)=y;
        y = y + sy;
        err = err - dx;
        if err < 0
            x = x + sx;
            err = err + dy;
        end
    end
end
xx(end+1)=x2; yy(end+1)=y2;
x=xx; y=yy;
end
