require 'nn';
require 'nngraph';
--require 'logger'
require 'utils.OneHot'

MixingGraph = 
MemoryModule = 
RepresentationModule = 
InfereenceGraph = require 'models.inference_module';
data_loader = require 'utils.data_loader';

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Memory Network')
cmd:text()
cmd:text('Options')
cmd:text()
-- Data Loading parameters
cmd:option('-data','/Users/david/Documents/MemoryNetwork/data/data.txt','Data directory for ')
cmd:option('-recompute_tensors',0,'Recompute data tensors ( has to be 1 for first run)')
cmd:option()
-- Model parameters
cmd:option('-model','lstm','lstm, gru or mem_net')
cmd:option('-num_mem',2,'number of memory units')
cmd:option('-mem_size',3,'dimension of the memory unit')
cmd:option('-feature_dim',nil,'dimension of the vocabulary embedding')
cmd:option('-voc_size',30,'dimension of the vocabulary')
--[[
cmd:option('-',,)

cmd:option('-')
cmd:option('-')
cmd:option('-')
]]-- 

-- Loading data

data_loader = require 'utils.data_loader';
QUICKLOAD = false

if QUICKLOAD then
    x = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/sample.t7")
    y = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/label.t7")
    voc = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7")
    index = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7_index")
else 
    input_file = "/Users/david/Documents/MemoryNetwork/preprocessing/output.txt"
    out_vocab_file = "/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7"
    out_tensor_file = "/Users/david/Documents/MemoryNetwork/output_lua/data.t7"
    voc = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7")
    index = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7_index")
    x, y,f = data_loader.text_to_tensor(input_file,out_vocab_file,out_tensor_file)
end


-- parse input params
NUM_MEM = 10
VOCAB_SIZE = 58
N_ITERATIONS = 1000
LEARNING_RATE = 1e-5

--------------  Network Creation

-- Incoding input
local rep = RepresentationModule.create_network(VOCAB_SIZE)
-- print(rep:forward(x[1]):size())
-- Setting memory
mem_mod = MemoryModule.create_network(NUM_MEM,VOCAB_SIZE)
-- Infering on memory 
o_mod = InferenceGraph.create_network(VOCAB_SIZE,3*VOCAB_SIZE)
-- Inferinf on Response
r_mod = InferenceGraph.create_network(VOCAB_SIZE,3*VOCAB_SIZE)

criterion = nn.MarginRankingCriterion(0.1)

--------------  Support Function

function print_memory_table(x_ind,fact,num_mem)
    local sup_memory = {}
    local temp1 = -1
    for i=1,f[1]:size(1) do
        local temp = math.floor(fact[x_ind][i]/num_mem)
        if temp ~= temp1 and temp ~= 0 then 
            table.insert(sup_memory,temp)
            temp1 = temp
        end
    end
    return sup_memory
end

function get_facts_memory(x_ind,fact,num_mem)
    local sup_memory = {}
    local temp1 = -1
    for i=1,f[1]:size(1) do
        local temp = math.floor(fact[x_ind][i]/num_mem)
        if temp ~= temp1 and temp ~= 0 then 
            table.insert(sup_memory,temp)
            temp1 = temp
        end
    end
    return sup_memory
end

function gradUpdate(mlp, x, memory, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, memory)
   local gradCriterion = criterion:backward(pred, memory)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
end

-------------- 


for i=1,N_ITERATIONS do
    local indice = torch.random(1,x:size(1))-- picking up a sentence in the training set
    local x_i, y_i = x[indice], y[indice]
    local m_i = get_facts_memory(indice,f,NUM_MEM)
    
    -- Going Through the Network 
    local xrepr = rep:forward(x_i)
    local yrepr = rep:forward(y_i)
    local a = mem_mod:forward(xrepr) -- a is the table of memories

    --
    local j = torch.random(1,NUM_MEM) -- picking up a memory
    local o_label = o_mod:forward{xrepr, a[m_i[1]]}
    if m_i[1] == j then
    else 
        local o_score = o_mod:forward{xrepr,a[j]}
        print(o_label)
        print(o_score)
        local resized_pred = nn.JoinTable(1):forward{o_label, o_score}
        local err = criterion:forward(resized_pred,1)
        local gradCriterion = criterion:backward(resized_pred, 1)
        local aaa = nn.Reshape(2,1):forward(nn.JoinTable(1):forward(gradCriterion))
        print(aaa:size())
        o_mod:zeroGradParameters()
        o_mod:backward({xrepr,a[j]},aaa:transpose(1,2))
    end
end

print("Answer ")

-- local score_2 = r_mod:forward{xrepr,yrepr}
-- print(score_2)