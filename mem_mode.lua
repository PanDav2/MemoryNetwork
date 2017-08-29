-- debug.getregistry()['MemoryModule']
require 'nn'
local MemoryModule, parent = torch.class('MemoryModule','nn.Module')

function MemoryModule:__init(NUM_MEM,MEM_SIZE,VOCAB_SIZE)
    parent.__init(self)
    self.memory = {}
    for i=1,NUM_MEM do
        self.memory = {[i] = torch.Tensor(MEM_SIZE,VOCAB_SIZE):fill(0)}
    end
end

function MemoryModule:updateOutput(input)
    return self.memory
end