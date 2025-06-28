function positionLine = myDrawLine()
% Hàm vẽ đường thẳng nằm ngang với khả năng dịch chuyển vùng chọn trước khi xác nhận.

disp('Nhấp vào một điểm để vẽ đường thẳng nằm ngang.');
[x, y] = ginput(1); % Chọn một điểm

% Làm tròn tọa độ
x = round(x);
y = round(y);

% Xác định giới hạn ảnh
x_limits = xlim; % Giới hạn trục x
x1 = round(x_limits(1));
x2 = round(x_limits(2));

% Tạo ROI dạng đường thẳng
roiLine = drawline('Position', [x1, y; x2, y], 'Color', 'r');

% Chờ người dùng dịch chuyển và xác nhận (nhấn Enter)
disp('Di chuyển đường thẳng nếu cần, sau đó nhấn Enter để xác nhận.');
wait(roiLine);

% Lấy vị trí cuối cùng của đường thẳng
positionLine = round(roiLine.Position);
end
