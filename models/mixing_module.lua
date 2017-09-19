MixingGraph = {}

function MixingGraph.create_network()
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local repr_x = nn.Reshape(1,-1,false)(nn.Sum(1)(inputs[1]))
    local repr_y = nn.Reshape(1,-1,false)(nn.Sum(1)(inputs[2]))
    local sum = nn.CAddTable(1)({repr_x,repr_y})
    --local sum = nn.CAddTable(1):forward{inputs[1],inputs[2]}
    table.insert(outputs, sum)
    return nn.gModule(inputs,outputs)
end


return MixingGraph