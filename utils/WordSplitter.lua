word_splitter = {}

function word_splitter.text_to_tensor(in_textfile)
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
end
return word_splitter
