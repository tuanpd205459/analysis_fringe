
function [images, num_images] = load_folder_images(image_folder)
    % Đọc các file ảnh từ thư mục
    image_files = dir(fullfile(image_folder, '*.bmp')); % Đọc các file ảnh .jpg
    num_images = length(image_files);

    % Khởi tạo mảng để lưu trữ các ảnh
    images = cell(1, num_images);

    % Đọc các ảnh từ thư mục
    for i = 1:num_images
        filename = fullfile(image_folder, image_files(i).name);
        images{i} = imread(filename);
    end
end
