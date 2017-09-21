require 'nn';
require 'nngraph';
require 'utils.OneHot';
require 'math';
require 'optim';
require 'gnuplot';
gnuplot.setterm("dumb")


MixingGraph = require 'models.mixing_module';
MemoryModule = require 'models.memory_module';
RepresentationModule = require 'models.representation_module';
InferenceGraph = require 'models.inference_module';
data_loader = require 'utils.data_loader';

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Memory Network')
cmd:text()
cmd:text('Options')
cmd:text()
-- Data Loading parameters
cmd:option('-generate_dataset',0,'generate the torch data representation')

-- Model parameters
cmd:option('-num_mem',10,'number of memory units')
cmd:option('-feature_dim',174,'dimension of the vocabulary embedding')
cmd:option('-voc_size',58,'dimension of the vocabulary')
cmd:option('-weight_initialization',"random","define the weight initialization")
-- Training parameters
cmd:option('-print_every',10, "print loss every x reps")
cmd:option('-learning_rate',1e-5, "the learning rate")
cmd:option('-iterations',1000, "Number of iterations through the network")
cmd:option('-num_epoch',100,"number of epochs during training")
cmd:option('-val_percentage',0.1,"amount of data using to create the training set")
cmd:option('-plot',0,"Plot the loss curves after training (1 to plot)")
    
opt = cmd:parse(arg)

data_loader = require 'utils.data_loader';

local N_ITERATIONS = 1000
local LEARNING_RATE = 1e-5

if opt.generate_dataset == 0 then
    x = torch.load("output_lua/sample.t7")
    y = torch.load("output_lua/label.t7")
    voc = torch.load("output_lua/vocab.t7")
    index = torch.load("output_lua/vocab.t7_index")
    f = torch.load("output_lua/fact_tensor.t7")
else 
    input_file = "preprocessing/output.txt"
    out_vocab_file = "output_lua/vocab.t7"
    out_tensor_file = "output_lua/data.t7"
    out_label_file = "output_lua/label.t7"
    index = torch.load("output_lua/vocab.t7_index")
    x, y, f = data_loader.text_to_tensor(input_file,out_vocab_file,out_tensor_file,out_label_file)
end

--------------  Network Creation

-- Incoding input
rep = RepresentationModule.create_network(opt.voc_size)
-- print(rep:forward(x[1]):size())
-- Setting memory
mem_mod = MemoryModule.create_network(opt.num_mem,opt.voc_size)
-- Infering on memory 
o_mod = InferenceGraph.create_network(opt.voc_size,opt.feature_dim)
o_mod_2 = o_mod:clone()
-- Infering on Response
r_mod = InferenceGraph.create_network(opt.voc_size,opt.feature_dim)
r_mod_2 = r_mod:clone()

criterion = nn.MarginRankingCriterion(0.1)
criterion_2 = criterion:clone()


if opt.weight_initialization ~= "random" then
    -- Setting up all weights to 1
    for k, node in pairs(o_mod.forwardnodes) do
        if node.data.annotations.name == "inference_proj_1" then
            local t1 = node.data.module.weight:fill(opt.weight_initialization)
            --print()
        end
        if node.data.annotations.name == "inference_proj_2" then
            local t2 = node.data.module.weight:fill(opt.weight_initialization)
        end
    end
end

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

local loss = {}
local total_loss = 0
local total_loss = 0

logger = optim.Logger('accuracy.log')
logger:setNames{'o_err', 'r_err'}


for ii = 1, opt.num_epoch do 
    for i=1,opt.iterations do
        local indice = torch.random(1,x:size(1))-- picking up a sentence in the training set
        local x_i, y_i = x[indice], y[indice]
        local m_i = get_facts_memory(indice,f,opt.num_mem)        
        -- Going Through the Network 
        local xrepr = rep:forward(x_i)
        local a = mem_mod:forward(xrepr) -- a is the table of memories
        -- sample a word 
        local _word = torch.Tensor(1):fill(torch.random(1,58))
        while _word[1] == y_i[1] do  -- there has to be a better way to do that
            _word = torch.Tensor(1):fill(torch.random(1,58))
        end
        -- sample a supporting memory
        local j = torch.random(1,opt.num_mem)
        while m_i[1] == j do 
            j = torch.random(1,opt.num_mem)
        end         

        -- Infering on memory

        local o_label = o_mod:forward{xrepr, a[m_i[1]]}
        
        local o_score = o_mod_2:forward{xrepr,a[j]}

        local resized_pred = nn.JoinTable(1):forward{o_label, o_score}
        local o_err = criterion:forward(resized_pred,1)
        local gradCriterion = criterion:backward(resized_pred, 1)
        local crit_false = nn.Reshape(1,1,false):forward(gradCriterion[2])
        local crit_true = nn.Reshape(1,1,false):forward(gradCriterion[1])

        -- updating o_module
        o_mod:zeroGradParameters()
        o_mod:backward({xrepr,m_i[1]},crit_true)
        o_mod_2:backward({xrepr,a[j]},crit_false)
        
        -- Infering on R

        local best_score = 0
        local max_mem
        for jj=1,opt.num_mem do
            local o_score = o_mod:forward{xrepr,a[jj]}
            if o_score[1][1] > best_score then
                max_mem = jj
            end
        end

        local yrepr = rep:forward(y_i)
        local _word_repr = rep:forward(_word) 
        local r_score = r_mod:forward{a[max_mem],_word_repr}
        local r_label = r_mod_2:forward{a[max_mem], yrepr}
        local resized_pred = nn.JoinTable(1):forward{r_label, r_score}
        local r_err = criterion_2:forward(resized_pred,1)
        local gradCriterion = criterion_2:backward(resized_pred, 1)
        local crit_false = nn.Reshape(1,1,false):forward(gradCriterion[2])
        local crit_true = nn.Reshape(1,1,false):forward(gradCriterion[1])


        -- -- updating r_module
        r_mod:zeroGradParameters()
        r_mod:backward({a[max_mem],_word_repr},crit_false)
        r_mod_2:backward({a[max_mem],yrepr},crit_true)


        loss[#loss] = o_err 
        total_loss = total_loss + o_err + r_err
        if i % opt.print_every == 0 then
            -- print("o_err : ".. o_err)
            -- print("r_err : ".. r_err)
            logger:add{o_err, r_err}
        end
        
    end
    print("epoch "..ii.." average_loss : "..total_loss/opt.iterations)
    total_loss = 0
end 
if opt.plot ~= 0 then
    logger:style{'+-', '+-'}
    logger:plot()
end
