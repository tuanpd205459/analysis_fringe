function crossLine = myCrossSection(inputSurface, positionLine)
    x1 = positionLine(1,1);
    y1 = positionLine(1,2);
    x2 = positionLine(2,1);
    y2 = positionLine(2,2);

    % Kiểm tra kích thước ảnh
    [rows, cols] = size(inputSurface);

    % Xử lý đường ngang (cùng hàng, khác cột)
    if y1 == y2  
        % Giới hạn x trong phạm vi ảnh
        x1 = max(1, min(cols, round(x1)));
        x2 = max(1, min(cols, round(x2)));

        % Lấy dữ liệu trực tiếp từ hàng y1
        crossLine = inputSurface(y1, x1:x2);

    % Xử lý đường thẳng đứng (cùng cột, khác hàng)
    elseif x1 == x2  
        % Giới hạn y trong phạm vi ảnh
        y1 = max(1, min(rows, round(y1)));
        y2 = max(1, min(rows, round(y2)));

        % Lấy dữ liệu trực tiếp từ cột x1
        crossLine = inputSurface(y1:y2, x1);

    else
        error('Đường không thẳng hàng hoặc cột, vui lòng chọn lại.');
    end
end
