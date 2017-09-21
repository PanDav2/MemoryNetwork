InferenceGraph = {}

function InferenceGraph.create_network(voc_size, feature_dim)
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    -- Projection des deux entrées sur les différents modules
    local repr_x = nn.Reshape(1,-1,false)(nn.Sum(1)(inputs[1]))
    local repr_y = nn.Reshape(1,-1,false)(nn.Sum(1)(inputs[2]))
    local lin1 = nn.Linear(voc_size,feature_dim)(repr_x):annotate{name="inference_proj_1", style="filled", fillcolor = "yellow"}
    local lin2 = nn.Linear(voc_size,feature_dim)(repr_y):annotate{name="inference_proj_2", style="filled", fillcolor = "yellow"}
    local prod = nn.MM(false,true){lin1,lin2}:annotate{name= "inference_product", style="filled", fillcolor = "yellow"}
    -- Outputs
    table.insert(outputs,prod)
    return nn.gModule(inputs, outputs)
end

return InferenceGraph