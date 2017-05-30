require 'nn'
require 'torch'
require 'nngraph'


base_model = {}

local function build_memory(params,input,context,time)
	local hid = {}
	hid[0] = input
	local sharelist = {}
	sharelist[1] = {}

	local AInc 
end

function base_model.i_module(input)
	inputs =  {}
	table.insert(inputs,nn.Identity()()) -- Entry feature map
	table.insert(outputs,nn.Identity()())
	return nn.gModule(inputs,outputs)
end

function base_model.g_module(number_of_memories)
	table.insert(inputs,nn.Identity()())
	table.insert(outputs,nn.Identity()())
	return nn.gModule(inputs,outputs)
end

function base_model.o_module()

end

function base_model.r_module()

end


return base_model