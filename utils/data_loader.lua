INPUT_FILE = "/Users/david/Documents/MemoryNetwork/data/data.txt"
OUT_VOCAB_FILE = "/Users/david/Documents/MemoryNetwork/output_lua/vocab.t7"
OUT_TENSOR_FILE = "/Users/david/Documents/MemoryNetwork/output_lua/sample.t7"
OUT_LABEL_FILE = "/Users/david/Documents/MemoryNetwork/output_lua/label.t7"

local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.split(input_file, sep,debug)
    --[[
        This method seperate the training samples from the label (response)
    ]]
    sep = sep or '|'
    debug = debug or 0
    local x = {}
    local y = {}
    local w 
    f = assert(io.open(input_file,"r"))
    repeat 
    rawdata= f:read()
    if rawdata then 
        w = rawdata:split("|") 
        x[#x+1] = w[1]
        y[#y+1] = w[2]
    end
    until not rawdata
    return x,y
end

function DataLoader.create_vocabulary(sample_tab,label_tab,out_vocab_file)
    --[[
        This method create the vocabulary resources based on a input text file
    ]]
    out_vocab_file = out_vocab_file or OUT_VOCAB_FILE
    local rawdata
    local max_sent_len = 0
    local sent_count = 0
    local unordered = {}
    for i=1,#sample_tab do 
        -- Writing on Sample File
        rawdata = sample_tab[i]
        for k,word in pairs(rawdata:split(" ")) do 
            word=word:lower()
            if not unordered[word] then unordered[word] = true end
        end
        sent_len = #rawdata:split(" ")
        if sent_len > max_sent_len then max_sent_len=sent_len end
        -- Writing on Label File
        rawdata = label_tab[i]
        for k,word in pairs(rawdata:split(" ")) do 
            word=word:lower()
            if not unordered[word] then unordered[word] = true end
        end
        sent_len = #rawdata:split(" ")
        if sent_len > max_sent_len then max_sent_len=sent_len end
        sent_count = sent_count + 1        
    end
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for word in pairs(unordered) do ordered[#ordered + 1] = word end
    table.sort(ordered)
    -- invert `ordered` to create the word->int mapping
    local vocab_mapping = {}
    local index_mapping = {}
    for i, word in ipairs(ordered) do
        index_mapping[i+1] = word
        vocab_mapping[word] = i+1
    end
    local  out_index_file = out_vocab_file..'_index'
    print('saving ' .. out_vocab_file)
    torch.save(out_vocab_file, vocab_mapping)
    print('saving its index in '.. out_index_file )
    torch.save(out_index_file, index_mapping)
    return {sent_count,vocab_mapping,max_sent_len,index_mapping}
end

function DataLoader.create_tensor(sent_count,vocab_mapping,max_sent_len,input_tab,tensor_file)
    --[[
        Create a tensor of word indice based on the input text
    ]]
    local tensor_file = tensor_file or OUT_TENSOR_FILE
    local data = torch.ByteTensor(sent_count,max_sent_len):fill(1)
    local currline = 1
    local rawdata
    -------- Writing in the tensor file 
    for i=1,#input_tab do 
        rawdata = input_tab[i]:lower()
        for k,word in pairs(rawdata:split(" ")) do 
            data[{currline,k}] = vocab_mapping[word:lower()]
        end
        currline = currline + 1
    end
    -- save output preprocessed files
    print('saving ' .. tensor_file)
    torch.save(tensor_file, data)
    return data
end

function DataLoader.text_to_tensor(input_file, out_vocab_file, out_label_tensor_file,out_sample_tensor_file)
    --[[
        Run the text to tensor process
    ]]
    local input_file = input_file or INPUT_FILE
    local out_vocab_file = out_vocab_file or OUT_VOCAB_FILE
    local out_label_tensor_file = out_label_tensor_file or OUT_LABEL_FILE
    local out_sample_tensor_file = out_sample_tensor_file or OUT_TENSOR_FILE

    local input,label = DataLoader.split(input_file)
    local output_create_vocab = DataLoader.create_vocabulary(input,label)
    
    local sent_count = output_create_vocab[1]
    local vocab_mapping = output_create_vocab[2]
    local max_sent_len = output_create_vocab[3]
    local index_mapping = output_create_vocab[4]

    print('putting sample tensor into '.. out_sample_tensor_file..'...')
    sample = DataLoader.create_tensor(sent_count,vocab_mapping,max_sent_len,input,out_sample_tensor_file)
    print('putting label tensor into '.. out_label_tensor_file..'...')
    label = DataLoader.create_tensor(sent_count,vocab_mapping,max_sent_len,label,out_label_tensor_file)
    return sample,label
end


function DataLoader.count_table_elements(t)
    --[[
        Simply count the amount of elements in a table
    ]]
    local count = 0
    for _ in pairs(t) do 
        count = count+1
    end 
    return count
end

return DataLoader