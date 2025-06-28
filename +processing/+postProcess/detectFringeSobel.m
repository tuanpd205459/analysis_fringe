function direction = detectFringeSobel(image)
    % Tính gradient theo hướng x và y
    Gx = imfilter(double(image), fspecial('sobel')');
    Gy = imfilter(double(image), fspecial('sobel'));

    % Tính tổng độ lớn gradient theo mỗi hướng
    sumX = sum(abs(Gx(:)));
    sumY = sum(abs(Gy(:)));

    % So sánh độ mạnh biên độ theo hướng x và y
    if sumX > sumY
        direction = 'vân dọc'; 
    else
        direction = 'vân ngang'; 
    end
end
