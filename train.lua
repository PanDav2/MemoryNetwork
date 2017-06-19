require 'torch'
require 'nn'
require 'nngraph'
require 'lfs'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Memory Network')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-input_file','data/data.txt','Data File. Should point to the actual file we want to use with the data')
cmd:option('-out_vocab_file','out/','data directory. Should contain the file input.txt with input data')
cmd:option('-out_tensor_file','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
cmd:option('-sep','|','Separating character between the sample input and the sample label (output)')
cmd:option('')

local data_loader = require 'data_loader.lua'