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

return BatchLoader