require 'nn'
require 'nngraph'
require 'logger'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Memory Network')
cmd:text()
cmd:text('Options')
cmd:text()
-- Model parameters
cmd:option('-num_mem','2','number of memory units')
cmd:option('-mem_size','3','dimension of the memory unit')
cmd:option('-feature_dim',,'dimension of the vocabulary embedding')
cmd:option('-voc_size',30,'dimension of the vocabulary')
cmd:option('-',,)


cmd:option('-')
cmd:option('-')
cmd:option('-')

-- parse input params
opt = cmd:parse(arg)
