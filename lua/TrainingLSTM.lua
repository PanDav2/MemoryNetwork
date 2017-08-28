
require 'nn'
require 'nngraph'
require 'optim'

data_loader = require 'utils.data_loader';
QUICKLOAD = true

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
    x, y = data_loader.text_to_tensor(input_file,out_vocab_file,out_tensor_file)
end

debug.getregistry()["OneHot"] = nil
OneHot, parent = torch.class('OneHot', 'nn.Module')

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self._eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input)
  self.output:resize(input:size(1), self.outputSize):zero()
  if self._eye == nil then self._eye = torch.eye(self.outputSize) end
  self._eye = self._eye:float()
  local longInput = input:long()
  self.output:copy(self._eye:index(1, longInput))
  return self.output
end

require 'nn';
require 'nngraph';
LSTM = require 'models.lstm';
--require 'utils.OneHot';
model_utils = require 'utils.model_utils';

RNN_SIZE = 10
NUM_LAYERS = 3
DROPOUT = 0
SEQ_LENGTH = 54
BATCH_SIZE = 1
VOCAB_SIZE = data_loader.count_table_elements(voc)+2
GRAD_CLIP = 5
LEARNING_RATE = 2e-3
LEARNING_RATE_DECAY = 0.97
LEARNING_RATE_DECAY_AFTER = 10
DECAY_RATE = 0.95
MAX_EPOCH = 50

BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(x,y,batch_size,seq_length)
    local self = {}
    setmetatable(self,BatchLoader)
    -- self.batches is a table of tensor
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.ix = 1
    
    self.x_batches = x:view(BATCH_SIZE,-1):split(SEQ_LENGTH,2) -- #rows = #batches
    self.nbatches = #self.x_batches
    
    self.y_batches = y:view(BATCH_SIZE,-1):split(SEQ_LENGTH,2) -- #rows = #batches
    self.y_nbatches = #self.y_batches
    print(#self.x_batches)
    print(#self.y_batches)
    assert(#self.x_batches == #self.y_batches)
    
    -- lets try to be helpful here
    if self.nbatches < 50 then
        print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
    end
    
    collectgarbage()
    return self 
end

function BatchLoader:next_batch()
    if self.nbatches < self.ix then
        self.ix = 1 -- cycling through the batch
    end
    local x = self.x_batches[self.ix]
    local y = self.y_batches[self.ix]
    self.ix = self.ix + 1
    return x,y 
end

B = BatchLoader.create(x,y,BATCH_SIZE,SEQ_LENGTH)

B.x_batches[1]



LSTM = require 'models.lstm';
protos = {}
protos.rnn = LSTM.lstm(VOCAB_SIZE, RNN_SIZE, NUM_LAYERS, DROPOUT)
protos.criterion = nn.ClassNLLCriterion()

-- graph.dot(protos.rnn.fg, 'rnn')

function prepro(x,y)
    local x = x:transpose(1,2):contiguous()
    local y = y:transpose(1,2):contiguous()
    return x,y
end

init_state = {}
for L=1,NUM_LAYERS do
    local h_init = torch.zeros(BATCH_SIZE, RNN_SIZE)
    table.insert(init_state,h_init:clone())
    table.insert(init_state,h_init:clone()) -- because LSTM
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
print("initialized parameters")
-- initialization
params:uniform(-0.08,0.08) -- small uniform numbers
print("parameters uniformed")
-- initialize the LSTM forget gates with slightly higher biases to encorage remembering in the beginning
for layer_idx = 1, NUM_LAYERS do
    for _, node in ipairs(protos.rnn.forwardnodes) do
        if node.data.annotations.name  == "i2h".. layer_idx then
            print('setting forget gate biases to 1 in LSTM layer '.. layer_idx) 
            -- the gates are in order i,f,o,g so f is the 2nd block of weights
            node.data.module.bias[{{RNN_SIZE+1,2*RNN_SIZE}}]:fill(1.0)
        end
    end
end

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto,SEQ_LENGTH, not proto.parameters)
end

function clone_list(tensor_list, zeto_too)
    -- utility function. todo : move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

local init_state_global = clone_list(init_state)
function feval(opti_params)
    --[[if opti_params ~= params then 
        params:copy(opti_params)
    end ]]--
    grad_params:zero()
    ------------------ get minibatch -------------------
    local x,y = B:next_batch()
    local x,y = prepro(x,y)
    ------------------ forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {} --- softmax outputs
    local loss = 0
    for t = 1,SEQ_LENGTH do
        -- print("iterations : "..t)
        clones.rnn[t]:training()  -- make sure we are in correct mode (this is cheap, sets flag)
        -- print(unpack(rnn_state[t-1]))
        local lst = clones.rnn[t]:forward{x[t],unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t],y[t])
    end
    loss = loss / SEQ_LENGTH
    ------------------ backward pass -------------------
    -- initialize radient at time t to be zeros (there's no influence from future)
    local drnn_state = {[SEQ_LENGTH] = clone_list(init_state,true)} -- true also zeros the clones
    for t = SEQ_LENGTH,1,-1 do
        -- backprop through loss, and softmax / linear
        local doutput_t = clones.criterion[t]:backward(predictions[t],y[t])
        table.insert(drnn_state[t],doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k> 1 then --k == 1 is gradient on x, which we don't need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. 
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------ backward pass -------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- does this need to be a clone ?
    -- grad_params:div(SEQ_LENGTH)
    -- clip gradient element wise
    grad_params:clamp(-GRAD_CLIP,GRAD_CLIP)
    return loss, grad_params
end

-- feval()

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learning_rate = LEARNING_RATE, alpha= DECAY_RATE}
local iterations = MAX_EPOCH * B.nbatches
local iterations_per_epoch = MAX_EPOCH*B.nbatches
local loss0 = nil

-- Optimization starts here
for i =1, iterations do
    local epoch = i / B.nbatches
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval,params,optim_state)
    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss
    print(train_losses[i])
    -- exponential learning rate decay
    if i % B.nbatches == 0 and LEARNING_RATE_DECAY < 1 then
        if epoch >= LEARNING_RATE_DECAY_AFTER then
            local decay_factor = LEARNING_RATE_DECAY
            optim_state.learning_rate = optim_state.learning_rate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learning_rate)
        end
    end
    
    -- every now and then or on last iteration
    if i % 10 == 0 then collectgarbage() end
    
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end





function tensor_to_table(tens,ind)
    ind = ind or 1
    local sentence = {}
    for i=1,tens:size(2) do
        table.insert(sentence,index[tens[{ind,i}]])
    end
    return sentence
end

function table_to_string(sentence)
    local s = ''
    for k,w in pairs(sentence) do
        s = s..' '.. w
    end
    return s
end

local tt = tensor_to_table(x)
local ttt = table_to_string(tt)
print(#tt)
print(ttt)

TEMP_SAMPLING = 0



------------------ loading the first element of the bash  -------------------
-- Using the first element of training set to test empirically if the model is working

local tt = tensor_to_table(x)
local ttt = table_to_string(tt)
seed_text = tt
len = #tt


------------------ Computing predictions (log probabilities at each timestep) -------------------
-- 

protos.rnn:evaluate()
-- local current_state
current_state = {}
for L = 1, NUM_LAYERS do
    -- c and h for all layers
    local h_init = torch.zeros(1,RNN_SIZE):double()
    table.insert(current_state,h_init:clone())
    table.insert(current_state,h_init:clone())
end
state_size = #current_state

-- do a few seeded timesteps
if len > 0 then
    -- print('seeding with '.. seed_text)
    print('-----------------------')
    for k,w in pairs(tt) do
        print('"'..w..'"')
        prev_word = torch.Tensor{voc[w]}
        local lst = protos.rnn:forward{prev_word, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1, state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end    
else
    print('please add some seeding text')
end

------------------ samapling / argmaxing over the log probabilities at each timestep -------------------
--

-- start sampling / argmaxing
for i=1, 5 do
    -- log probabilities from the previous timestep
    if TEMP_SAMPLING == 0 then
        -- use argmax 
        local _, prev_word_ = prediction:max(2)
        prev_word = prev_word_:resize(1)
    else
        -- use sampling
        prediction:div(TEMP) -- scale by temperatrue
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_word = torch.multinomial(probs:float(), 1):resize(1):float()
    end
    -- forward the nn for next word
    local lst = protos.rnn:forward{prev_word, unpack(current_state)}
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    prediction = lst[#lst] -- last element holds the log probs
    print(index[prev_word[1]])
end

torch.Tensor{voc['billy.']}
print(voc['billy.'])