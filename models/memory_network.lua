require 'nn';
require 'nngraph'
require 'OneHot'

local IModule, parent = torch.class('IModule', 'nn.Module')

function 

function IModule:__init(num_mem,input_size)
	parent.__init(self)
    self.input_size = input_size or 54
    self.mem_size = input_size/num_mem
    assert(isint(mem_size), "input_size/num_mem is not integer. input_size "..input_size .. " num_mem "..num_mem)
    local voc_size = dl.count_table_elements(voc)
    o = OneHot(voc_size+1) 
    self.mem_nn = nn.Sequential()
    -- I Module
    self.mem_nn:add(o)
    -- G Module
    self.mem_nn:add(nn.Reshape(num_mem,input_size/num_mem,voc_size+1))
    self.mem_nn:add(nn.SplitTable(1))
    self.mem_nn:add(nn.ParallelTable():add(a[1]):add(a[2]))
    self.mem_nn:add(nn.JoinTable())
    return mem_nn
end


function IModule:forward(input)
	self.mem_nn:forward(input)
end