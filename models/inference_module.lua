InferenceModule, parent = torch.class('InferenceModule','nn.Module')

function InferenceModule:__init(voc_size, feature_dim)
    parent.__init(self)
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    -----  
    local lin1 = nn.Linear(voc_size,feature_dim)(inputs[1])
    local lin2 = nn.Linear(voc_size,feature_dim)(inputs[2])
    table.insert(outputs,lin1)
    table.insert(outputs,lin2)
    self.mlp = nn.gModule(inputs, outputs) 
end

function InferenceModule:updateOutput(input)
    local ind = input[3] or 1
    local input1 = input[1][ind]:clone()
    local input2 = input[2]:clone()
    local i4 = self.mlp:forward{input1,input2}
    local ll = i4[2]:transpose(1,2)
    local lll = i4[1]:reshape(1,90)
    local glo = lll*ll
    return glo
end