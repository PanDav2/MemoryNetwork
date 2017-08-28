others = {}

function others.create_index(end,start)
    local a
    local i = start or 0
    a=  torch.Tensor(end):apply(function() i=i+1;return i end)    
    return a
end

return others