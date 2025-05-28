clc,clear, close all; 
addpath('./data')
Img_Original = imread('untitled2.png');
%% === Chuyển ảnh về xám và nhị phân ===
grayImg = rgb2gray(Img_Original);
img = im2double(grayImg);
figure; imshow(img); title('Ảnh xám');

%% === Nhị phân hóa bằng ngưỡng Otsu ===
thresh = graythresh(grayImg);
BW_Original = ~imbinarize(grayImg, thresh);  % Đảo màu: nền = 0, đối tượng = 1
figure; imshow(BW_Original); title('Ảnh nhị phân ban đầu');


changing = 1;
[rows, columns] = size(BW_Original);
BW_Thinned = BW_Original;
while changing
    BW_Del = ones(rows, columns); % Reset lại mỗi vòng lặp
    changing = 0;
    % Step 1
    for i=2:rows-1
        for j = 2:columns-1
            P = [BW_Thinned(i,j) BW_Thinned(i-1,j) BW_Thinned(i-1,j+1) BW_Thinned(i,j+1) BW_Thinned(i+1,j+1) BW_Thinned(i+1,j) BW_Thinned(i+1,j-1) BW_Thinned(i,j-1) BW_Thinned(i-1,j-1) BW_Thinned(i-1,j)];
            if (BW_Thinned(i,j) == 1 &&  sum(P(2:end-1))<=6 && sum(P(2:end-1)) >=2 && P(2)*P(4)*P(6)==0 && P(4)*P(6)*P(8)==0)
                A = 0;
                for k = 2:size(P,2)-1
                    if P(k) == 0 && P(k+1)==1
                        A = A+1;
                    end
                end
                if (A==1)
                    BW_Del(i,j)=0;
                    changing = 1;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned.*BW_Del;
    BW_Del = ones(rows, columns); % Reset lại trước bước 2
    % Step 2 
    for i=2:rows-1
        for j = 2:columns-1
            P = [BW_Thinned(i,j) BW_Thinned(i-1,j) BW_Thinned(i-1,j+1) BW_Thinned(i,j+1) BW_Thinned(i+1,j+1) BW_Thinned(i+1,j) BW_Thinned(i+1,j-1) BW_Thinned(i,j-1) BW_Thinned(i-1,j-1) BW_Thinned(i-1,j)];
            if (BW_Thinned(i,j) == 1 && sum(P(2:end-1))<=6 && sum(P(2:end-1)) >=2 && P(2)*P(4)*P(8)==0 && P(2)*P(6)*P(8)==0)
                A = 0;
                for k = 2:size(P,2)-1
                    if P(k) == 0 && P(k+1)==1
                        A = A+1;
                    end
                end
                if (A==1)
                    BW_Del(i,j)=0;
                    changing = 1;
                end
            end
        end
    end
    BW_Thinned = BW_Thinned.*BW_Del;
end

figure;
imshow(BW_Original);
figure;
imshowpair(BW_Original, BW_Thinned, 'blend');
title('Overlay skeleton on original using imshowpair');
figure;
imshow(BW_Thinned);
