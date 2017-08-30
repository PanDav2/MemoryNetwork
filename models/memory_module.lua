MemoryModule, parent = torch.class('MemoryModule','nn.Module')

function MemoryModule:__init(NUM_MEM,MEM_SIZE,VOCAB_SIZE)
    parent.__init(self)
    self._num_mem = NUM_MEM or 
    self._mem_size = MEM_SIZE
    self._memory = {}
    for i=1,NUM_MEM do table.insert(self._memory,torch.Tensor(MEM_SIZE,VOCAB_SIZE):fill(0)) end
end

function MemoryModule:updateOutput(input)
    -- Replace the index memory with the input it as received
    assert(input:size(2) == self._memory[1]:size(2), "input size and memory size are differents")
    local input = input:clone()
    local loaded_mem = 0
    for i=1,#self._memory do
       for j=1,self._mem_size do
            if j + loaded_mem > input:size(1) then
                break
            end
            self._memory[i][{j}] = input[{j + loaded_mem,{}}]
        end
        loaded_mem = loaded_mem + self._mem_size
    end
    return self._memory
end

function MemoryModule:getIndex(index)
    return self._memory[index]
end

function MemoryModule:getMemorySize()
    return #self._memory
end

function MemoryModule:getMemory()
    return nn.JoinTable(1):forward(self._memory)
end