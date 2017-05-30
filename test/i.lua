require 'nn'
require 'nngraph'


i_component = {}

function i_component.build_i(input)
	local inputs = {}
	local outputs = {}
	table.insert(inputs,nn.Identity()())
	table.insert(outputs,nn.Identity()())
	return nn.gModule(inputs,outputs)
end

return i_component
