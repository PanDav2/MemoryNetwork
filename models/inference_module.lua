InferenceGraph = {}

function InferenceGraph.create_module(voc_size, feature_dim)
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    -- Projection des deux entrées sur les différents modules
    local lin1 = nn.Linear(voc_size,feature_dim)(inputs[1]):annotate{name="proj_1"}
    local lin2 = nn.Linear(voc_size,feature_dim)(inputs[2]):annotate{name="proj_2"}
    local prod = nn.MM(true){lin1,lin2}
    -- Transposing : lin1.data.module.weight:transpose(1,2)*lin2.data.module.weight
    -- Outputs 
    table.insert(outputs,prod)    
    return nn.gModule(inputs, outputs)
end

return InferenceGraph