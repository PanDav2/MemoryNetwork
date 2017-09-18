MemoryModule = {}

function MemoryModule.create_network(num_mem,longuest_sentence_size,debug)--longest_sentence,voc_size,num_mem)
    local inputs = {}
    local outputs = {}
    table.insert(inputs, nn.Identity()())
    -- 
    local reshaped_input =  nn.Reshape(1,-1)(inputs[1])
    local mem_proj = nn.SplitTable(1)(reshaped_input):annotate{name="mem_proj"}
    -- Creating memory adressage :
    local mem_size =  math.floor(longuest_sentence_size/num_mem)
    if debug then print("num_mem = "..num_mem) print("mem_size = "..mem_size) end
    for i=1,num_mem do
        local mem_mapping = nn.NarrowTable(1+(i-1)*mem_size,mem_size)(mem_proj)
        table.insert(outputs, nn.JoinTable(1)(mem_mapping)) -- Joining to get a table of tensors
    end 
    return nn.gModule(inputs, outputs)
end