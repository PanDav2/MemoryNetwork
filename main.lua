require 'nn'
require 'nngraph'
--require 'logger'
require 'models.memory_network'
require 'utils.OneHot'
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

-- parse input params
opt = cmd:parse(arg)
if not opt.feature_dim then
	opt.feature_dim = 3*opt.voc_size
end

if recompute_tensors then
    input_file = "/Users/david/Documents/MemoryNetwork/preprocessing/output.txt"
    out_vocab_file = "/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7"
    out_tensor_file = "/Users/david/Documents/MemoryNetwork/output_lua/data.t7"
    voc = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7")
    index = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7_index")
    x, y = data_loader.text_to_tensor(input_file,out_vocab_file,out_tensor_file)
else
	x = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/sample.t7")
    y = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/label.t7")
    voc = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7")
    index = torch.load("/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7_index") 
end

if opt.model == 'lstm' then
	print('ERROR. lstm not implemented yet')
	os.exit()
else
	print('all good')
end