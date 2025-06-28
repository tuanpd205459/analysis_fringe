function wrappedPhase = processFourier(hologram,selectRegion_HCN, maxIntensity)
% processFourier: thực hiện các phép biến đổi Fourier và chọn vùng bậc +1

if nargin < 2 || isempty(maxIntensity) || isempty(selectRegion_HCN)
    selectRegion_HCN = true;
    maxIntensity = 100000;
end


%% Chuyển ảnh hologram sang grayscale
hologramGray = myConvGrayScale(hologram);  % Chuyển đổi ảnh sang grayscale

[numRows,numCols] = size(hologramGray);  % Lấy kích thước ảnh (chiều cao và chiều rộng)

%% Biến đổi Fourier và chọn ROI

fourierTransform = fftshift(fft2(hologramGray));  % Thực hiện biến đổi Fourier và dịch chuyển zero-frequency về trung tâm

% Hiển thị ảnh đã biến đổi Fourier
figure;
imshow(abs(fourierTransform), [1, maxIntensity]);
% imshow(log(abs(fourierTransform)), []);
title('Biến đổi Fourier của Hologram');

%% Xác định +1 của phổ
% Vẽ hình chữ nhật để chọn ROI
if(selectRegion_HCN)
    [pos, xRec, yRec, widthRec, heightRec] = myDrawRec();
    fprintf("Kich thuoc hinh vuong: %.2fx %.4f\nDien tich HCN: %.2f \n", widthRec,heightRec, widthRec*heightRec);
    % Xacs dinh tam bac 1
    umax = xRec + widthRec/2;
    vmax = yRec + heightRec/2;
    % Tính góc θx (đơn vị radian)
    u0 = numRows/2+1;
    v0 = numCols/2 +1;
    delta_xy = 3.45*1e-6;
    lambda = 633*1e-9;
    tilt_angleX = asin(abs(u0 - umax) * lambda / (numRows * delta_xy));
    tilt_angleY = asin(abs(v0 - vmax) * lambda / (numCols * delta_xy));

    % Đổi sang độ nếu cần
    theta_x_deg = rad2deg(tilt_angleX);
    theta_y_deg = rad2deg(tilt_angleY);


    
    % Hiển thị giá trị góc nghiêng - Radian
    disp(['Gia tri goc nghieng Theta_x va Theta_y la: (', num2str(tilt_angleX), ', ', num2str(tilt_angleY), ') - radian']);

    % Hiển thị giá trị góc nghiêng - Độ
    disp(['Gia tri goc nghieng Theta_x va Theta_y la: (', num2str(theta_x_deg), ', ', num2str(theta_y_deg), ') - Độ']);

    % Trích xuất nội dung ROI (bậc 1)
    roiContent = fourierTransform(yRec:yRec + heightRec - 1, xRec:xRec + widthRec - 1);
else
    [centerCir_X, centerCir_Y, radiusCir] = myDrawCir();
    fprintf("Ban kinh duong tron: %.2f\nDien tich duong tron: %.2f \n", radiusCir, 3.14*radiusCir*radiusCir);
    roiContent = fourierTransform(centerCir_Y-radiusCir:centerCir_Y+radiusCir-1, centerCir_X-radiusCir:centerCir_X+radiusCir-1);
end


%% Xác định +1 của phổ
% Vẽ hình chữ nhật để chọn ROI
if (selectRegion_HCN == false)
    newFourierTransform = zeros(size(fourierTransform));
    % Tính toán vị trí trung tâm mới cho newFourierTransform
    centerX = round(numRows / 2);
    centerY = round(numCols / 2);

    % Tạo vùng để chèn ROI hình tròn vào trung tâm của newFourierTransform
    newFourierTransform(centerX - radiusCir:centerX + radiusCir - 1, ...
        centerY - radiusCir:centerY + radiusCir - 1) = roiContent;
    % Tạo mặt nạ hình tròn để giữ lại các phần tử bên trong vòng tròn
    [xGrid, yGrid] = meshgrid(1:numCols, 1:numRows);
    circleMask = sqrt((xGrid - centerY).^2 + (yGrid - centerX).^2) <= radiusCir;

    % Áp dụng mặt nạ để giữ các phần tử bên trong vòng tròn, đặt phần tử ngoài vòng tròn thành 0
    newFourierTransform(~circleMask) = 0;

end

if(selectRegion_HCN == true)
    [numRows,numCols] = size(fourierTransform);
    newFourierTransform = zeros(size(fourierTransform));
    % Tính toán vị trí trung tâm mới
    centerX = round((numRows - heightRec) / 2);  % Tọa độ x của tâm
    centerY = round((numCols - widthRec) / 2); % Tọa độ y của tâm

    newFourierTransform(centerX:centerX + heightRec - 1, centerY:centerY + widthRec - 1) = ...
        roiContent;
end

finalPhase = ifft2(ifftshift(newFourierTransform));
wrappedPhase = angle(finalPhase);

% Hiển thị newFourierTransform
figure;
imshow(abs(newFourierTransform), [1, maxIntensity]);
title('Anh sau khi dich vao tam');

wrappedPhase =wrappedPhase';
figure();
mesh(wrappedPhase);
title('Pha chua Unwrapping');



end

%% hàm vẽ hình chữ nhật
function [pos, xRec, yRec, widthRec, heightRec] = myDrawRec()
%     roi = drawrectangle();  % Rectangle

% Vẽ hình chữ nhật để chọn ROI
roi = drawrectangle();

% Lấy tọa độ tâm ban đầu
centerRec = round([roi.Position(1) + roi.Position(3)/2, roi.Position(2) + roi.Position(4)/2]);

% Vẽ dấu cộng tại tâm
hold on;
centerMarker = plot(centerRec(1), centerRec(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

% Thiết lập callback để cập nhật tâm trong quá trình di chuyển hình chữ nhật
addlistener(roi, 'MovingROI', @(src, evt) updateCenterRectangle(src, centerMarker));


wait(roi);  % Double-click to confirm the ROI

% Get the position of the rectangle [x, y, width, height]
pos = round(roi.Position);
xRec = pos(1);
yRec = pos(2);
widthRec = pos(3);
heightRec = pos(4);
end


% Hàm cập nhật tâm khi di chuyển ROI
function updateCenterRectangle(roi, centerMarker)
% Cập nhật vị trí của dấu cộng theo tọa độ tâm mới
centerX = roi.Position(1) + roi.Position(3)/2;
centerY = roi.Position(2) + roi.Position(4)/2;
centerMarker.XData = centerX;
centerMarker.YData = centerY;
drawnow;
end

%% hàm vẽ đường tròn
function [centerCir_X, centerCir_Y, radiusCir] = myDrawCir()
%roi = drawcircle();

% Vẽ vòng tròn để chọn ROI
roi = drawcircle();

% Lấy tọa độ ban đầu của tâm và bán kính
centerCir = round(roi.Center);

% Vẽ dấu cộng tại tâm
hold on;
centerMarker = plot(centerCir(1), centerCir(2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);

hold off;

% Thiết lập callback để cập nhật tâm trong lúc di chuyển vòng tròn
addlistener(roi, 'MovingROI', @(src, evt) updateCenter(src, centerMarker));

% Hàm cập nhật tâm trong quá trình di chuyển ROI



wait(roi);
centerCir_X = round(roi.Center(1));  % Tọa độ x của trung tâm hình tròn
centerCir_Y = round(roi.Center(2));  % Tọa độ y của trung tâm hình tròn
radiusCir = round(roi.Radius);
end

function updateCenter(roi, centerMarker)
% Cập nhật vị trí của dấu cộng theo tọa độ tâm mới
centerMarker.XData = roi.Center(1);
centerMarker.YData = roi.Center(2);
drawnow;
end

%% Hàm chuyển sang grayscale
function output = myConvGrayScale(inputImage)
    if size(inputImage, 3) > 1
        inputImage = rgb2gray(inputImage);
    end
    output = double(inputImage); 
end

