clc;clear;close all;
% tương tác giữa script và app
app = app1_fringe_detection_backup4_6();
uiwait(app.UIFigure);        % Đợi đến khi nhấn "Export"

% Lấy các biến từ app sau khi người dùng nhấn "Export"
img = app.grayImg;
skeleton = app.Skeleton;
lambda = app.lambda;
surface = app.recons_surface;

% Xử lý tiếp
disp("Lambda:");
disp(lambda);

imshow(img); title("Gray Image");

% Xoá app
delete(app);
