import torch
import torch.nn as nn

class PartI_JobEncoder(nn.Module):
    def __init__(self, scalar_input_dim=1, embedding_dim=64, hidden_dim=128, rnn_type='RNN', num_rnn_layers=1):
        super(PartI_JobEncoder, self).__init__()
        
        self.scalar_input_dim = scalar_input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()
        self.num_rnn_layers = num_rnn_layers
        
        self.embedding = nn.Linear(scalar_input_dim, embedding_dim)
        
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_rnn_layers,
                              batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}. Choose from 'RNN', 'LSTM', or 'GRU'.")
        
    def forward(self, job_processing_times, h_0=None, c_0=None):
        """
            Forward pass for Part I Job Encoder.

            Args:
                job_processing_times (Tensor): Tensor of processing times for a batch of jobs.
                                             Shape: (batch_size, num_jobs, num_machines, scalar_input_dim)
                h_0 (Tensor, optional): Initial hidden state for RNN/GRU.
                                        Shape: (num_rnn_layers * num_directions, batch_size * num_jobs, hidden_dim)
                                        Defaults to zeros if None.
                c_0 (Tensor, optional): Initial cell state for LSTM.
                                        Shape: (num_rnn_layers * num_directions, batch_size * num_jobs, hidden_dim)
                                        Defaults to zeros if None. (Only for LSTM)

            Returns:
                Tensor: Encoded job vectors p_i.
                        Shape: (batch_size, num_jobs, hidden_dim)
        """
        batch_size = job_processing_times.size(0)
        num_jobs = job_processing_times.size(1)
        num_machines = job_processing_times.size(2)

        reshaped_times = job_processing_times.view(batch_size * num_jobs, num_machines, self.scalar_input_dim)
        
        embedded_seq = self.embedding(reshaped_times)
        
        current_batch_size = embedded_seq.size(0)

        if h_0 is None and self.rnn_type in ['RNN', 'GRU']:
            h_0 = torch.zeros(self.num_rnn_layers, current_batch_size, self.hidden_dim).to(job_processing_times.device)
        
        if self.rnn_type == 'LSTM':
            if h_0 is None:
                h_0 = torch.zeros(self.num_rnn_layers, current_batch_size, self.hidden_dim).to(job_processing_times.device)
            if c_0 is None:
                c_0 = torch.zeros(self.num_rnn_layers, current_batch_size, self.hidden_dim).to(job_processing_times.device)
            rnn_output, (h_n, c_n) = self.rnn(embedded_seq, (h_0, c_0))
        else:
            rnn_output, h_n = self.rnn(embedded_seq, h_0)
            
        p_i_flat = h_n[-1, :, :] # Shape: (batch_size * num_jobs, hidden_dim)

        p_i = p_i_flat.view(batch_size, num_jobs, self.hidden_dim)

        return p_i

class PartII_MachineIntegration(nn.Module):
    def __init__(self, p_vector_dim, num_machines_scalar_dim=1, m_embedding_dim=32, fc_hidden_dim=128, output_dim=64):
        """
        Part II of PFSPNet: Integrates machine count 'm' with job vectors 'p_i'.

        Args:
            p_vector_dim (int): Dimension of input job vector p_i (from Part I).
            num_machines_scalar_dim (int): Dimension of scalar machine count 'm' (typically 1).
            m_embedding_dim (int): Dimension of the embedded machine count vector m_tilde.
            fc_hidden_dim (int): Dimension of the hidden layer in the FC network (optional, can be direct to output_dim).
                                 For simplicity, we can have one FC layer mapping to output_dim.
            output_dim (int): Dimension of the output vector p_tilde_i.
        """
        super(PartII_MachineIntegration, self).__init__()
        self.p_vector_dim = p_vector_dim
        self.m_embedding_dim = m_embedding_dim

        self.machine_embedding = nn.Linear(num_machines_scalar_dim, m_embedding_dim)

        self.fc_layer = nn.Linear(m_embedding_dim + p_vector_dim, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, p_vectors, num_machines_scalar):
        """
            Forward pass for Part II.

            Args:
                p_vectors (Tensor): Batch of p_vectors from Part I.
                                    Shape: (batch_size, num_jobs, p_vector_dim)
                num_machines_scalar (Tensor): Batch of scalar machine counts.
                                              Shape: (batch_size,) or (batch_size, 1).

            Returns:
                Tensor: Output vectors p_tilde_vectors.
                        Shape: (batch_size, num_jobs, output_dim)
        """
        batch_size = p_vectors.size(0)
        num_jobs = p_vectors.size(1)

        if num_machines_scalar.dim() == 1:
            num_machines_scalar = num_machines_scalar.unsqueeze(1)

        m_tilde = self.machine_embedding(num_machines_scalar)

        m_tilde_expanded = m_tilde.unsqueeze(1).expand(-1, num_jobs, -1)

        concatenated_input = torch.cat((m_tilde_expanded, p_vectors), dim=2)

        h_tilde = self.fc_layer(concatenated_input)

        p_tilde_vectors = self.relu(h_tilde)

        return p_tilde_vectors   # shape: [batch_size, num_jobs, output_dim]

class PartIII_Convolutional(nn.Module):
    def __init__(self, p_tilde_dim, conv_out_channels, conv_kernel_size=3, conv_padding='same'):
        """
            Part III of PFSPNet: 1D Convolution over job vectors p_tilde_i.

            Args:
                p_tilde_dim (int): Dimension of input vectors p_tilde_i (from Part II).
                                This is 'in_channels' for Conv1d.
                conv_out_channels (int): Dimension of output vectors p_bar_i.
                                        This is 'out_channels' for Conv1d.
                conv_kernel_size (int): Kernel size for Conv1d.
                conv_padding (str or int): Padding for Conv1d. 'same' tries to keep seq length.
        """
        super(PartIII_Convolutional, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=p_tilde_dim,
                                out_channels=conv_out_channels,
                                kernel_size=conv_kernel_size,
                                padding=conv_padding) 
        
        self.relu = nn.ReLU()

    def forward(self, p_tilde_vectors_sequence):
        """
            Forward pass for Part III.

            Args:
                p_tilde_vectors_sequence (Tensor): Sequence of p_tilde vectors from Part II.
                                                 Shape: (batch_size, num_jobs, p_tilde_dim)

            Returns:
                Tensor: Output vectors p_bar.
                        Shape: (batch_size, num_jobs, conv_out_channels)
        """
        # Input shape: (batch_size, num_jobs, p_tilde_dim)
        # Conv1d expects: (batch_size, in_channels, seq_len), so (batch_size, p_tilde_dim, num_jobs)
        input_for_conv = p_tilde_vectors_sequence.transpose(1, 2)
        
        conv_output = self.conv1d(input_for_conv)

        activated_output = self.relu(conv_output)
        
        # Output shape from conv: (batch_size, conv_out_channels, num_jobs)
        # Transpose back to: (batch_size, num_jobs, conv_out_channels)
        p_bar_vectors = activated_output.transpose(1, 2)

        return p_bar_vectors

class PFSPNetEncoder(nn.Module):
    def __init__(self, part1_args, part2_args, part3_args):
        """
        Complete PFSPNet Encoder (Part I + Part II + Part III).

        Args:
            part1_args (dict): Arguments for PartI_JobEncoder.
                               e.g., {'scalar_input_dim':1, 'embedding_dim':64, 'hidden_dim':128, 'rnn_type':'RNN'}
            part2_args (dict): Arguments for PartII_MachineIntegration.
                               e.g., {'p_vector_dim':128, 'm_embedding_dim':32, 'output_dim':64}
                               (p_vector_dim must match part1's hidden_dim)
            part3_args (dict): Arguments for PartIII_Convolutional.
                               e.g., {'p_tilde_dim':64, 'conv_out_channels':128, 'conv_kernel_size':3}
                               (p_tilde_dim must match part2's output_dim)
        """
        super(PFSPNetEncoder, self).__init__()

        self.part1_encoder = PartI_JobEncoder(**part1_args)
        
        # Ensure dimensions match between parts
        if part1_args['hidden_dim'] != part2_args['p_vector_dim']:
            raise ValueError("Output dimension of Part I (hidden_dim) must match "
                             "input dimension of Part II (p_vector_dim).")
        self.part2_integrator = PartII_MachineIntegration(**part2_args)

        if part2_args['output_dim'] != part3_args['p_tilde_dim']:
            raise ValueError("Output dimension of Part II (output_dim) must match "
                             "input dimension of Part III (p_tilde_dim).")
        self.part3_convoluter = PartIII_Convolutional(**part3_args)

    def forward(self, instance_processing_times, num_machines_scalar):
        """
        Forward pass for the complete PFSPNet Encoder.

        Args:
            instance_processing_times (Tensor): Processing times for a batch of instances.
                                                Shape: (batch_size, num_jobs, num_machines_in_job_seq, scalar_input_dim)
            num_machines_scalar (Tensor or float): Scalar number of machines 'm' for each instance in the batch.
                                                   Shape: (batch_size,) or (batch_size, 1).

        Returns:
            Tensor: Final encoded job vectors p_bar.
                    Shape: (batch_size, num_jobs, part3_conv_out_channels)
        """
        # Part I: Encode each job's processing time sequence
        # Input: (batch_size, num_jobs, num_machines_in_job_seq, 1)
        # Output p_vectors: (batch_size, num_jobs, part1_hidden_dim)
        p_vectors = self.part1_encoder(instance_processing_times)

        # Part II: Integrate machine count 'm'
        # Input p_vectors: (batch_size, num_jobs, part1_hidden_dim)
        # Input num_machines_scalar: e.g., tensor([5., 4., 5.])
        # Output p_tilde_vectors: (batch_size, num_jobs, part2_output_dim)
        p_tilde_vectors = self.part2_integrator(p_vectors, num_machines_scalar)

        # Part III: 1D Convolution over the sequence of p_tilde_vectors
        # Input p_tilde_vectors: (batch_size, num_jobs, part2_output_dim)
        # Output p_bar_vectors: (batch_size, num_jobs, part3_conv_out_channels)
        p_bar_vectors = self.part3_convoluter(p_tilde_vectors)

        return p_bar_vectors

"""
if __name__ == '__main__':
    batch_size = 3
    num_jobs_n = 10
    num_machines_m_val = 5

    part1_embedding_dim = 32
    part1_hidden_dim = 64
    part1_rnn_type = 'RNN'

    part2_p_vector_dim = part1_hidden_dim
    part2_m_embedding_dim = 16
    part2_fc_output_dim = 48

    part3_p_tilde_dim = part2_fc_output_dim
    part3_conv_out_channels = 96
    part3_conv_kernel_size = 3
    part3_conv_padding = 'same'

    proc_times = torch.rand(batch_size, num_jobs_n, num_machines_m_val, 1)
    
    part1_scalar_input_dim = proc_times.shape[-1]
    
    part1_args = {
        'scalar_input_dim': part1_scalar_input_dim,
        'embedding_dim': part1_embedding_dim,
        'hidden_dim': part1_hidden_dim,
        'rnn_type': part1_rnn_type
    }
    part2_args = {
        'p_vector_dim': part2_p_vector_dim,
        'm_embedding_dim': part2_m_embedding_dim,
        'output_dim': part2_fc_output_dim
    }
    part3_args = {
        'p_tilde_dim': part3_p_tilde_dim,
        'conv_out_channels': part3_conv_out_channels,
        'conv_kernel_size': part3_conv_kernel_size,
        'conv_padding': part3_conv_padding
    }

    pfsp_encoder = PFSPNetEncoder(part1_args, part2_args, part3_args)

    
    sample_m_scalar = torch.tensor([float(num_machines_m_val)] * batch_size)

    print(f"Input instance processing times shape: {proc_times.shape}")
    print("-" * 40)
    
    print(f"Input machine count m: {sample_m_scalar}")
    print(f"Input machine count m shape: {sample_m_scalar.shape}")
    print("-" * 40)

    final_encoded_vectors = pfsp_encoder(proc_times, sample_m_scalar)
    print("-" * 40)

    print(f"Output p_bar vectors shape: {final_encoded_vectors.shape}")
    print(f"Expected: (batch_size, num_jobs, part3_conv_out_channels) = ({batch_size}, {num_jobs_n}, {part3_conv_out_channels})")
    print("-" * 40)

    part1_module = PartI_JobEncoder(**part1_args)
    p_vecs = part1_module(proc_times)
    print(f"Part I output shape: {p_vecs.shape}")

    part2_module = PartII_MachineIntegration(**part2_args)
    p_tilde_vecs = part2_module(p_vecs, sample_m_scalar)
    print(f"Part II output shape: {p_tilde_vecs.shape}")
    
    part3_module = PartIII_Convolutional(**part3_args)
    p_bar_vecs = part3_module(p_tilde_vecs)
    print(f"Part III output shape: {p_bar_vecs.shape}")
 """