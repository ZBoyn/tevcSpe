import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class JobProcessingTimeEncoderForDecoder(nn.Module):
    def __init__(self, scalar_input_dim=1, embedding_dim=64, hidden_dim=128, rnn_type='RNN', num_rnn_layers=1):
        super(JobProcessingTimeEncoderForDecoder, self).__init__()
        self.embedding = nn.Linear(scalar_input_dim, embedding_dim)

        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers
        self.hidden_dim = hidden_dim
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type: {}".format(rnn_type))
    
    def forward(self, proc_times_seq, initial_hidden_state=None, initial_cell_state=None):
        batch_size = proc_times_seq.size(0)
        embedded_seq = self.embedding(proc_times_seq) # (1, num_machines, embedding_dim)

        if self.rnn_type == 'LSTM':
            if initial_hidden_state is None:
                h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(proc_times_seq.device)
            else:
                h0 = initial_hidden_state
            if initial_cell_state is None:
                c0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(proc_times_seq.device)
            else:
                c0 = initial_cell_state
            _, (h_n, _) = self.rnn(embedded_seq, (h0, c0))
        else: # RNN / GRU
            if initial_hidden_state is None:
                h0 = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_dim).to(proc_times_seq.device)
            else:
                h0 = initial_hidden_state
            _, h_n = self.rnn(embedded_seq, h0)
            
        return h_n[-1, :, :]

class DecoderStep1Stage(nn.Module):
    # 第一阶段的解码器，用于处理 P_pi-1,j 的编码器 (RNN1) 和机器数量 m 的嵌入
    def __init__(self, pt_encoder_args, m_embedding_dim, rnn1_output_dim, di_output_dim, fc_hidden_dims=None):
        super(DecoderStep1Stage, self).__init__()
        # 用于处理 P_pi-1,j 的编码器 (RNN1)
        self.pt_encoder = JobProcessingTimeEncoderForDecoder(**pt_encoder_args)
        self.machine_embedding = nn.Linear(1, m_embedding_dim)

        fc_input_dim = rnn1_output_dim + m_embedding_dim
        layers = []
        if fc_hidden_dims:
            for h_dim in fc_hidden_dims:
                layers.append(nn.Linear(fc_input_dim, h_dim))
                layers.append(nn.ReLU())
                fc_input_dim = h_dim
        layers.append(nn.Linear(fc_input_dim, di_output_dim))
        layers.append(nn.ReLU())
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, prev_job_proc_times, ptr_h_state, ptr_c_state, num_machines_scalar):
        
        rnn1_output = self.pt_encoder(prev_job_proc_times, 
                                      initial_hidden_state=ptr_h_state, 
                                      initial_cell_state=ptr_c_state) # (1, rnn1_output_dim)

        m_embedded = self.machine_embedding(num_machines_scalar) # (1, m_embedding_dim)

        concatenated = torch.cat((rnn1_output, m_embedded), dim=1) # (1, rnn1_output_dim + m_embedding_dim)

        di_vector = self.fc_layers(concatenated) # (1, di_output_dim)
        return di_vector

class DecoderStep2Stage(nn.Module):
    """
        Implements Step 2 of the decoding network:
        Inputs d_i and d_{i-1}^* to the second decoding RNN (RNN2)
        Outputs d_i^* (hidden state of RNN2) and rnn_out_i^* (output of RNN2).
    """
    def __init__(self, di_input_dim, rnn2_hidden_dim, rnn_type='RNN', num_rnn_layers=1):
        """
        Args:
            di_input_dim (int): Dimension of d_i vector from Step 1.
            rnn2_hidden_dim (int): Hidden dimension of RNN2. This is also the dimension of d_i^*
                                   and rnn_out_i^*. The d_{i-1}^* used for concatenation
                                   will also have this dimension.
            rnn_type (str): Type of RNN ('RNN', 'LSTM', 'GRU').
            num_rnn_layers (int): Number of layers for RNN2.
        """
        super(DecoderStep2Stage, self).__init__()
        self.di_input_dim = di_input_dim
        self.rnn2_hidden_dim = rnn2_hidden_dim
        self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers

        rnn2_actual_input_dim = di_input_dim + rnn2_hidden_dim

        if rnn_type == 'RNN':
            self.rnn2 = nn.RNN(rnn2_actual_input_dim, rnn2_hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn2 = nn.LSTM(rnn2_actual_input_dim, rnn2_hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn2 = nn.GRU(rnn2_actual_input_dim, rnn2_hidden_dim, num_layers=num_rnn_layers, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type: {}".format(rnn_type))

    def forward(self, di_vector, prev_rnn2_h_state, prev_rnn2_c_state=None):
        """
        Args:
            di_vector (torch.Tensor): The d_i vector from DecoderStep1Stage.
                                      Shape: (batch_size, di_input_dim)
            prev_rnn2_h_state (torch.Tensor): The hidden state (h-part of d_{i-1}^*) from the
                                              previous step of this RNN2.
                                              Shape: (num_rnn_layers, batch_size, rnn2_hidden_dim)
            prev_rnn2_c_state (torch.Tensor, optional): The cell state (c-part of d_{i-1}^*) for LSTM,
                                                        from the previous step. Defaults to None.
                                                        Shape: (num_rnn_layers, batch_size, rnn2_hidden_dim)
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] or torch.Tensor]:
            - rnn_out_i_star (torch.Tensor): RNN output for the current step.
                                             Shape: (batch_size, rnn2_hidden_dim)
            - current_rnn2_state (Tuple or Tensor): The new hidden state d_i^*.
                - For LSTM: (current_rnn2_h_state, current_rnn2_c_state)
                - For RNN/GRU: current_rnn2_h_state
              Shapes are (num_rnn_layers, batch_size, rnn2_hidden_dim)
        """
        batch_size = di_vector.size(0)

        # d_{i-1}^* for concatenation is the hidden state from the last layer of RNN2 from the previous step
        # prev_rnn2_h_state has shape (num_rnn_layers, batch_size, rnn2_hidden_dim)
        d_star_for_concat = prev_rnn2_h_state[-1] # Shape: (batch_size, rnn2_hidden_dim)

        # Concatenate d_i and d_{i-1}^* (from prev_rnn2_h_state's last layer)
        concat_input = torch.cat((di_vector, d_star_for_concat), dim=1)
        # Shape: (batch_size, di_input_dim + rnn2_hidden_dim)

        # RNNs expect input of shape (batch_size, seq_len, input_features)
        # Here, we are processing one step at a time, so seq_len = 1.
        rnn_input_seq = concat_input.unsqueeze(1)
        # Shape: (batch_size, 1, di_input_dim + rnn2_hidden_dim)

        if self.rnn_type == 'LSTM':
            if prev_rnn2_c_state is None: # Should be provided if LSTM
                # This might happen at the very first step (i=0 for d_0^*), where it's initialized to zeros.
                prev_rnn2_c_state = torch.zeros_like(prev_rnn2_h_state).to(prev_rnn2_h_state.device)
            rnn_output_seq, (current_h_state, current_c_state) = self.rnn2(rnn_input_seq, (prev_rnn2_h_state, prev_rnn2_c_state))
            current_rnn2_state = (current_h_state, current_c_state)
        else: # RNN or GRU
            rnn_output_seq, current_h_state = self.rnn2(rnn_input_seq, prev_rnn2_h_state)
            current_rnn2_state = current_h_state

        # rnn_output_seq has shape (batch_size, 1, rnn2_hidden_dim)
        # We need rnn_out_i^* which is the output for the current single time step
        rnn_out_i_star = rnn_output_seq.squeeze(1) # Shape: (batch_size, rnn2_hidden_dim)

        return rnn_out_i_star, current_rnn2_state

class AttentionModule(nn.Module):
    def __init__(self, job_encoding_dim, rnn_output_dim, attention_hidden_dim):
        """
        Args:
            job_encoding_dim (int): 单个工件编码 P_tilde_j 的维度。
            rnn_output_dim (int): 来自DecoderStep2Stage的 rnn_out_i_star 的维度。
            attention_hidden_dim (int): 注意力机制内部MLP的隐藏维度。
        """
        super(AttentionModule, self).__init__()
        self.job_encoding_dim = job_encoding_dim
        self.rnn_output_dim = rnn_output_dim
        self.attention_hidden_dim = attention_hidden_dim

        concat_dim = job_encoding_dim + rnn_output_dim

        self.W1_layer = nn.Linear(concat_dim, attention_hidden_dim, bias=True)

        self.v_layer = nn.Linear(attention_hidden_dim, 1, bias=False)

    def forward(self, P_tilde_all_jobs, rnn_out_i_star):
        """
        Args:
            P_tilde_all_jobs (torch.Tensor): 所有工件的编码。
            rnn_out_i_star (torch.Tensor): 来自 DecoderStep2Stage 的输出。
                                              Shape: (batch_size, num_total_jobs, job_encoding_dim)
        Returns:
            torch.Tensor: 每个工件的注意力得分 (logits) u_i。
                          Shape: (batch_size, num_total_jobs)
        """
        batch_size, num_total_jobs, _ = P_tilde_all_jobs.shape

        #扩展 rnn_out_i_star 以便与每个工件的编码进行拼接
        # rnn_out_i_star shape: (batch_size, rnn_output_dim)
        # expanded_rnn_out shape: (batch_size, num_total_jobs, rnn_output_dim)
        expanded_rnn_out = rnn_out_i_star.unsqueeze(1).expand(-1, num_total_jobs, -1)

        # 为所有工件拼接 P_tilde_j 和 rnn_out_i_star
        # concat_features shape: (batch_size, num_total_jobs, job_encoding_dim + rnn_output_dim)
        concat_features = torch.cat((P_tilde_all_jobs, expanded_rnn_out), dim=2)

        # 应用 W1 和 tanh
        # W1_projected shape: (batch_size, num_total_jobs, attention_hidden_dim)
        W1_projected = self.W1_layer(concat_features)
        activated_W1 = torch.tanh(W1_projected)

        # 应用 v
        # scores_ui shape: (batch_size, num_total_jobs, 1)
        scores_ui = self.v_layer(activated_W1)

        # 压缩最后一个维度得到 (batch_size, num_total_jobs)
        return scores_ui.squeeze(-1)

class SoftmaxModule(nn.Module):
    def __init__(self, dim=-1):
        super(SoftmaxModule, self).__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.dim = dim

    def forward(self, scores_ui, mask=None):
        if mask is not None:
            if not isinstance(mask, torch.BoolTensor):
                mask = mask.bool()

            scores_ui_masked = scores_ui.clone()
            scores_ui_masked[~mask] = -float('inf')
            return self.softmax(scores_ui_masked)
        else:
            return self.softmax(scores_ui)

class PFSPNetDecoder(nn.Module):
    def __init__(self,
                 step1_pt_encoder_args, step1_m_embedding_dim, step1_di_output_dim,
                 step2_rnn2_hidden_dim, attention_job_encoding_dim, attention_hidden_dim,
                 step1_fc_hidden_dims=None,
                 step2_rnn_type='RNN', step2_num_rnn_layers=1,
                 ptr_dim=None,
                 ):
        super(PFSPNetDecoder, self).__init__()

        self.step1_rnn1_is_lstm = step1_pt_encoder_args['rnn_type'] == 'LSTM'
        self.step1_rnn1_num_layers = step1_pt_encoder_args.get('num_rnn_layers', 1)
        self.step1_rnn1_hidden_dim = step1_pt_encoder_args['hidden_dim']
        
        self.step2_rnn2_is_lstm = step2_rnn_type == 'LSTM'
        self.step2_num_rnn_layers = step2_num_rnn_layers
        self.step2_rnn2_hidden_dim = step2_rnn2_hidden_dim

        self.step1_stage = DecoderStep1Stage(
            pt_encoder_args=step1_pt_encoder_args,
            m_embedding_dim=step1_m_embedding_dim,
            rnn1_output_dim=self.step1_rnn1_hidden_dim,
            di_output_dim=step1_di_output_dim,
            fc_hidden_dims=step1_fc_hidden_dims
        )
        self.step2_stage = DecoderStep2Stage(
            di_input_dim=step1_di_output_dim,
            rnn2_hidden_dim=self.step2_rnn2_hidden_dim,
            rnn_type=step2_rnn_type,
            num_rnn_layers=self.step2_num_rnn_layers
        )
        self.attention_module = AttentionModule(
            job_encoding_dim=attention_job_encoding_dim, # 来自编码器的输出维度
            rnn_output_dim=self.step2_rnn2_hidden_dim,
            attention_hidden_dim=attention_hidden_dim
        )
        self.softmax_module = SoftmaxModule(dim=-1)

        self.initial_d0_star_h = nn.Parameter(torch.randn(self.step2_num_rnn_layers, 1, self.step2_rnn2_hidden_dim) * 0.1)
        if self.step2_rnn2_is_lstm:
            self.initial_d0_star_c = nn.Parameter(torch.randn(self.step2_num_rnn_layers, 1, self.step2_rnn2_hidden_dim) * 0.1)
        else:
            self.register_parameter('initial_d0_star_c', None)

        if ptr_dim is not None:
            self.ptr_raw_learnable = nn.Parameter(torch.randn(1, ptr_dim) * 0.1)
            self.ptr_h_projection = nn.Linear(ptr_dim, self.step1_rnn1_num_layers * self.step1_rnn1_hidden_dim)
            if self.step1_rnn1_is_lstm:
                self.ptr_c_projection = nn.Linear(ptr_dim, self.step1_rnn1_num_layers * self.step1_rnn1_hidden_dim)
            else:
                self.ptr_c_projection = None
        else:
            self.ptr_raw_learnable = None


    def forward(self,
                encoded_jobs,
                all_job_processing_times,
                num_machines_scalar,
                external_ptr_h_state=None,
                external_ptr_c_state=None,
                max_decode_len=None,
                teacher_forcing_ratio=0.0,
                target_sequence=None
               ):
        batch_size, num_total_jobs, _ = encoded_jobs.shape
        device = encoded_jobs.device
        num_machines = all_job_processing_times.size(2)

        if max_decode_len is None:
            max_decode_len = num_total_jobs

        outputs_indices = []
        outputs_log_probs = []

        ptr_h_for_step1, ptr_c_for_step1 = None, None
        if self.ptr_raw_learnable is not None:
            ptr_batch_expanded = self.ptr_raw_learnable.expand(batch_size, -1) # (B, ptr_dim)
            ptr_h_flat = self.ptr_h_projection(ptr_batch_expanded) # (B, L1*H1)
            ptr_h_for_step1 = ptr_h_flat.view(batch_size, self.step1_rnn1_num_layers, self.step1_rnn1_hidden_dim).permute(1,0,2).contiguous() # (L1,B,H1)
            if self.step1_rnn1_is_lstm:
                ptr_c_flat = self.ptr_c_projection(ptr_batch_expanded) # (B, L1*H1)
                ptr_c_for_step1 = ptr_c_flat.view(batch_size, self.step1_rnn1_num_layers, self.step1_rnn1_hidden_dim).permute(1,0,2).contiguous()
        elif external_ptr_h_state is not None:
            ptr_h_for_step1 = external_ptr_h_state
            ptr_c_for_step1 = external_ptr_c_state
        else:
            ptr_h_for_step1 = torch.zeros(self.step1_rnn1_num_layers, batch_size, self.step1_rnn1_hidden_dim, device=device)
            if self.step1_rnn1_is_lstm:
                ptr_c_for_step1 = torch.zeros(self.step1_rnn1_num_layers, batch_size, self.step1_rnn1_hidden_dim, device=device)
        
        current_P_prev_job = torch.zeros(batch_size, num_machines, 1, device=device)
        current_rnn2_h = self.initial_d0_star_h.expand(-1, batch_size, -1).contiguous()
        current_rnn2_c = None
        if self.step2_rnn2_is_lstm:
            current_rnn2_c = self.initial_d0_star_c.expand(-1, batch_size, -1).contiguous()

        job_availability_mask = torch.ones(batch_size, num_total_jobs, device=device, dtype=torch.bool)
        batch_indices = torch.arange(batch_size, device=device)

        for t in range(max_decode_len):
            # 1. DecoderStep1Stage
            
            di_vec = self.step1_stage(current_P_prev_job, ptr_h_for_step1, ptr_c_for_step1, num_machines_scalar.view(batch_size, 1))

            # 2. DecoderStep2Stage
            rnn_out_star, rnn2_state_next = self.step2_stage(di_vec, current_rnn2_h, current_rnn2_c)
            
            # 3. AttentionModule
            attn_scores = self.attention_module(encoded_jobs, rnn_out_star)
            
            # 4. SoftmaxModule
            probs = self.softmax_module(attn_scores, job_availability_mask) # (B, NumJobs)
            log_probs = torch.log(probs + 1e-9)
            outputs_log_probs.append(log_probs)

            chosen_job_idx = None
            if self.training and torch.rand(1).item() < teacher_forcing_ratio and target_sequence is not None:
                chosen_job_idx = target_sequence[:, t] # (B,)
            else:
                chosen_job_idx = torch.multinomial(probs, 1).squeeze(1)
                # chosen_job_idx = torch.argmax(probs, dim=1) # Greedy
            outputs_indices.append(chosen_job_idx.unsqueeze(1)) # (B,1)

            current_P_prev_job = all_job_processing_times[batch_indices, chosen_job_idx] # (B, NumMachines, 1)
            
            if self.step2_rnn2_is_lstm:
                current_rnn2_h, current_rnn2_c = rnn2_state_next
            else:
                current_rnn2_h = rnn2_state_next
            
            job_availability_mask[batch_indices, chosen_job_idx] = False
            
            if not job_availability_mask.any(dim=1).all() and t < max_decode_len -1 :
                pass


        final_indices = torch.cat(outputs_indices, dim=1)
        final_log_probs = torch.stack(outputs_log_probs, dim=1)

        return final_indices, final_log_probs, encoded_jobs