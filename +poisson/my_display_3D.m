function [ ] = my_display_3D( data )
%DISPLAY the 2D data
%  figure;
 mesh(data); %3D
colormap jet,shading interp
hold on
zmax=max(max(data));zmin=min(min(data));caxis([zmin,zmax])
h=colorbar;
set(get(h,'title'),'string','rad','Fontname','Times New Roman')
end