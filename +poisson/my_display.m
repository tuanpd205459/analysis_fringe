function [ ] = my_display( data )
%DISPLAY the 2D data
%  figure;
 imagesc(data); %2D
colormap jet,shading interp
hold on
zmax=max(max(data));zmin=min(min(data));caxis([zmin,zmax])
h=colorbar;
set(get(h,'title'),'string','rad','Fontname','Times New Roman')
end

