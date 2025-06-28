function [reconSurface, dimensional] = myConvertUnit(reconSurface)
    % Kiểm tra giá trị lớn nhất để xác định đơn vị
%     if (max(reconSurface(:))-min(reconSurface(:))) < 10
    if max(abs(reconSurface(:))) > 1000
        scaleFactor = 10^3; % Chuyển từ nanomet sang micromet
        dimensional = 'micromet';
    else
        scaleFactor = 1;  
        dimensional = 'nanomet';
    end

    % Chuyển đổi đơn vị
    reconSurface = reconSurface / scaleFactor;
end
