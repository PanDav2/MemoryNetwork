require 'nn'
require 'nngraph'
require 'inference_module'
require 'memory_module'


function create_network(SEQ_LENGTH,VOCAB_SIZE,MEM_SIZE,NUM_MEM)
    ------------------ Initialization -------------------
    SEQ_LENGTH = SEQ_LENGTH or 5
    MEM_SIZE = MEM_SIZE or 3
    NUM_MEM = NUM_MEM or 2
    VOCAB_SIZE = VOCAB_SIZE or 30
    ------------------ I Module -------------------
    local mem_net = nn.Sequential()
    local branch_net = nn.ConcatTable()
    local mlp = nn.Sequential()
    local net = nn.Parallel(1,1)
    for i=1,SEQ_LENGTH do
        net:add(OneHot(VOCAB_SIZE))
    end
    mlp:add(net)    
    ------------------ G Module -------------------    
    -- m = MemoryModule.new(NUM_MEM,MEM_SIZE,VOCAB_SIZE)
    local g_mod = MemoryModule.new(NUM_MEM,MEM_SIZE,VOCAB_SIZE)
    mlp:add(g_mod)
    ------------------ O Module -------------------    
    local o_mod = InferenceModule.new(VOCAB_SIZE,3*VOCAB_SIZE)
    branch_net:add(mlp)
    branch_net:add(nn.Identity())
    mem_net:add(branch_net)
    mem_net:add(o_mod)
    ------------------ R Module -------------------
    return mem_net
end