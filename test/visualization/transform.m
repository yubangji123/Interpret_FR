function ndata = transform(data,range,target_range)

    ndata = (data - range(1)) .* (target_range(2) - target_range(1))/(range(2) - range(1)) + target_range(1);
    

end