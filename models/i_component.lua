require 'nn'

local IComponent, parent = torch.class('IComponent','nn.Module')

function IComponent:__init(outputSize,voc_size)
	parent.__init(self)
	self.outputSize = outputSize
	self._eye = torch.eye(outputSize)
end

function IComponent:updateOutput(input)
	self.output:resize(input:size(1),self.outputSize):zero()
	if self._eye == nil then self._eye = torch.eye(self.outputSize):zero() end
	self._eye = self._eye:float()
	local longInput = input:long()
	self.output:copy(self._eye:index(1,longInput))
	return self.output
end