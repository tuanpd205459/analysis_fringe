function hologram = loadHologram(inputManual, filePath, folder_path)
% loadImgHologram - Đọc ảnh hologram từ file hoặc thư mục
%
% Syntax:
%   loadImgHologram(inputManual, filePath, folder_path)
%
% Description:
%   Hàm này đọc ảnh hologram từ đường dẫn cụ thể hoặc tự động lấy ảnh mới nhất trong thư mục.
%
% Input Arguments:
%   inputManual : logical
%       1 - Đọc file bằng tay từ `filePath`
%       0 - Tự động tìm file mới nhất trong `folder_path`
%
%   filePath : string
%       Đường dẫn đầy đủ đến file ảnh (chỉ dùng nếu input file bằng tay`).
%
%   folder_path : string
%       Đường dẫn đến thư mục chứa ảnh (chỉ dùng nếu input file tự động`).
%
% Example:
%   % Đọc ảnh từ đường dẫn cụ thể
%   loadImgHologram(1, 'C:\data\hologram.bmp', '')
%
%   % Tự động lấy ảnh mới nhất từ thư mục
%   loadImgHologram(0, '', 'C:\data')
%

if nargin < 2 || isempty(filePath)
    inputManual = 0;
    folder_path = 'C:\Users\admin\Máy tính\Lab thầy Tùng\Thi nghiem\6-12';
end
if nargin < 3 || isempty(folder_path)
    filePath = '8.bmp';
    inputManual = 1;
end

if inputManual       
    hologram = imread(filePath);  % Đọc ảnh từ file
else
    files = dir(fullfile(folder_path, '*.bmp')); 
    if ~isempty(files)
        [~, idx] = max([files.datenum]); 
        latest_file = files(idx).name;
        full_path = fullfile(folder_path, latest_file);
        hologram = imread(full_path);
        fprintf('Ảnh mới nhất là: %s\n', latest_file);
    else
        disp('Không có file ảnh nào trong thư mục.');
    end
end

end
