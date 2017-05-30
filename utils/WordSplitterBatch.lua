local WordSPlitterMinibatchLoader = {}
WordSPlitterMinibatchLoader.__index = WordSPlitterMinibatchLoader

function WordSPlitterMinibatchLoader.create(data_dir,batch_size,seq_length,split_fractions)
	local self = {}
	setmetatable(self,WordSPlitterMinibatchLoader)

	local input_file = path.join(data_dir,'input.txt')
	local vocab_file = path.join(data_dir,'vocab_t7')
	local tensor_file = path.join(data_dir,'data.t7')

	local run_prepro = true
	if run_prepro then
		print('one-time setup : preprocessing input text file '.. input_file .. '....')
		WordSPlitterMinibatchLoader.text_to_tensor(input_file)--, vocab_file, tensor_file)
	end

	print('loading data files...')
	local data = torch.load(tensor_file)
	self.vocab_mapping = torch.load(vocab_file)

	-- cut off the end so that it divides evenly
	local len = data:size(1)
	if len % (batch_size * seq_length) ~= 0 then
		print(' cutting off end of data so that the batches/sequences divide evenly')
		data = data:sub(1,batch_size * seq_length* math.floor(len/batch_size*seq_length))
	
 	-- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end	

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
end


--- **** STATIC methhod *** --- 
function WordSPlitterMinibatchLoader.text_to_tensor(in_textfile)--, out_vocabfile, out_tensorfile)
	local timer = torch.Timer()
	data_dir = "/Users/david/Documents/MemoryNetwork/utils"

	local input_file = path.join(data_dir,'input.txt')
	local vocab_file = path.join(data_dir,'vocab_t7')
	local tensor_file = path.join(data_dir,'data.t7')

	print('loading text file....')
	local rawdata
	local tot_len = 0
	local f = assert(io.open(in_textfile,"r"))

	-- Create vocabulary if it doesn't exist yet
	print('creating vocabulary mapping')
	local unordered = {}
	rawdata = f:read():lower()
	repeat
		for k,word in pairs(rawdata:split(" ")) do 
			word=word:lower()
			if not unordered[word] then unordered[word] = true end
		end
		tot_len = tot_len + #rawdata:split(" ")
		rawdata = f:read()
	until not rawdata
	f:close()
	-- sort into a table (i.e. keys become 1..N)
	local ordered = {}
	for char in pairs(unordered) do ordered[#ordered + 1] = char end
	table.sort(ordered)
	-- invert `ordered` to create the char->int mapping
	local vocab_mapping = {}
	for i, word in ipairs(ordered) do
		vocab_mapping[word] = i
	end
	print('saving ' .. vocab_file)
    torch.save(vocab_file, vocab_mapping)
	-- construct a tensor with all the data
	print('putting data into tensor...')
	local data = torch.ByteTensor(tot_len) -- store it into 1D first, then rearrange
	f = assert(io.open(in_textfile,"r"))
	local currlen = 0
	
	-------- Writing in the tensor file 
	rawdata = f:read():lower()
	repeat
		for k,word in pairs(rawdata:split(" ")) do 
			data[currlen+k] = vocab_mapping[word:lower()]
		end
		currlen = currlen + #rawdata:split(" ")
		rawdata = f:read()
	until not rawdata
	f:close()
	
	-- save output preprocessed files
    print('saving ' .. tensor_file)
    torch.save(tensor_file, data)
    return data
end