RepresentationModule = {}

function RepresentationModule.create_network(vocab_size) 
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    -- 
    local one_hot = OneHot.new(vocab_size)(inputs[1]) -- One Hot Encoding sub
    table.insert(outputs, one_hot)
    return nn.gModule(inputs, outputs)
end

return RepresentationModule