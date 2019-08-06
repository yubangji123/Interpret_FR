function rgb = showWithColorMap(data, range, map)
    if (~exist('map', 'var')), map = jet(256); end;
    ndata = transform(data,range,[0,256]);
    
    rgb = ind2rgb(uint8(ndata),map);
    %imshow(rgb);
    
end