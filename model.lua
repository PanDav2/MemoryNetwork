LSTM = {}


function LSTM.build_model(input_size,rnn_size,num_layers,dropout)
	dropout = dropout or 0

	-- there will be 2*n+1 inputs
	local inputs = {}
	table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
	for L=1,num_layers do
		table.insert(inputs, nn.Identity()()) -- prev_c[L]
		table.insert(inputs, nn.Identity()()) -- prev_h[L]
	end

	local x, input_size_L
	local outputs = {}

	for L = 1,n fo
		local prev_h = inputs[2*L+1]
		local prev_c = inputs[2*L]
		if L == 1 then
			x = OneHot(input_size)(inputs[1])
			input_size_L = input_size
		else
			x = outputs[(L-1)*2]
			if dropout > 1 then nn.Dropout(dropout)(x) end
			input_size_L = rnn_size
		end

		-- updating all memory at once for efficiency
		local i2h = nn.Linear(input_size_L, 4* rnn_size)(x):annotate{name='i2h_'..L}
		local h2h = nn.Linear(input_size_L, 4* rnn_size)(prev_h):annotate{name='h2h_'..L}
		local all_input_sums = nn.CAddTable()({i2h,h2h})

		local reshaped = nn.Reshape(4,rnn_size)(all_input_sums)
		local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
		-- decode the gates 
		local in_gate = nn.Sigmoid()(n1)
		local forget_gate = nn.Sigmoid()(n2)
		local out_gate = nn.Sigmoid()(n3)

		-- decode he wirte inputs
		local in_transform = nn.Tanh()(n4)
		-- perform the LSTM update
		local next_c = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_c}),
		nn.CMulTable()({in_gate, in_transform})
		})
		-- gated cells form the ouput
		local next_h = nn.CMulTable()({out_gate,nn.Tanh()(next_c)})

		table.insert(outputs,next_c)
		table.insert(outputs,next_h)

		-- set up the decoder
		local top_h = outputs[#outputs]
		if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
		local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
		local logsoft = nn.LogSoftMax()(proj)
		table.insert(outputs,logsoft)

		return nn.gModule(inputs,outputs)
end

return LSTM