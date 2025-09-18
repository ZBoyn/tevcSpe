from pfsp_decode import PFSPNetDecoder
from pfsp_encode import PFSPNetEncoder
import torch
import torch.nn as nn

class PFSPNet(nn.Module):
    def __init__(self, encoder_args, decoder_args):
        super(PFSPNet, self).__init__()
        self.encoder = PFSPNetEncoder(**encoder_args)
        self.decoder = PFSPNetDecoder(**decoder_args)
    
    def forward(self, 
                batch_instance_processing_times,  # (B, NumJobs, NumMachines, 1)
                batch_num_machines_scalar,        # (B, 1)
                external_ptr_h_state=None,  
                external_ptr_c_state=None,
                max_decode_len=None,
                teacher_forcing_ratio=0.0,
                target_sequence=None
                ):
        
        encoded_jobs_batched = self.encoder(batch_instance_processing_times, batch_num_machines_scalar)

        selected_indices, log_probs, encoded_jobs_batched = self.decoder(
            encoded_jobs_batched,
            all_job_processing_times=batch_instance_processing_times,
            num_machines_scalar=batch_num_machines_scalar,            
            external_ptr_h_state=external_ptr_h_state,
            external_ptr_c_state=external_ptr_c_state,
            max_decode_len=max_decode_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
            target_sequence=target_sequence
        )
         
        return selected_indices, log_probs, encoded_jobs_batched


class CriticNetwork(nn.Module):
    def __init__(self,
                 encoder_output_dim,       
                 conv_channels_list,       
                 conv_kernel_sizes_list,   
                 conv_strides_list=None,  
                 final_fc_hidden_dims=None
                ):
        super(CriticNetwork, self).__init__()

        if not isinstance(conv_channels_list, list) or len(conv_channels_list) != 3:
            print(f"Warning: Paper suggests 3 conv layers for critic. Received channels list: {conv_channels_list}")
        if not isinstance(conv_kernel_sizes_list, list) or len(conv_kernel_sizes_list) != 3:
            print(f"Warning: Paper suggests 3 conv layers for critic. Received kernel sizes list: {conv_kernel_sizes_list}")

        if conv_strides_list is None:
            conv_strides_list = [1] * len(conv_channels_list)
        elif not isinstance(conv_strides_list, list) or len(conv_strides_list) != 3:
             print(f"Warning: Paper suggests 3 conv layers for critic. Received strides list: {conv_strides_list}")


        if not (len(conv_channels_list) == len(conv_kernel_sizes_list) == len(conv_strides_list)):
            raise ValueError("conv_channels_list, conv_kernel_sizes_list, and conv_strides_list must have the same length.")

        conv_layers_seq = []
        current_channels = encoder_output_dim 

        for i in range(len(conv_channels_list)):
            out_channels = conv_channels_list[i]
            kernel_size = conv_kernel_sizes_list[i]
            stride = conv_strides_list[i]
            
            padding = (kernel_size - 1) // 2
            
            conv_layers_seq.append(nn.Conv1d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            conv_layers_seq.append(nn.ReLU())
            current_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers_seq)
        
        self.last_conv_output_channels = current_channels

        fc_layers_list = []
        input_dim_for_current_fc = self.last_conv_output_channels
        if final_fc_hidden_dims and isinstance(final_fc_hidden_dims, list):
            for hidden_dim in final_fc_hidden_dims:
                fc_layers_list.append(nn.Linear(input_dim_for_current_fc, hidden_dim))
                fc_layers_list.append(nn.ReLU())
                input_dim_for_current_fc = hidden_dim
        
        fc_layers_list.append(nn.Linear(input_dim_for_current_fc, 1))
        self.output_fc = nn.Sequential(*fc_layers_list)

    def forward(self, encoded_jobs):
        x = encoded_jobs.permute(0, 2, 1)
        
        x = self.conv_layers(x)
        x_summed = torch.sum(x, dim=2)
        
        baseline_value = self.output_fc(x_summed)
        
        return baseline_value

""" 
if __name__ == '__main__':
    batch_size = 4
    num_jobs = 20
    num_machines = 5
    
    part1_hidden_dim = 64
    part2_output_dim = 48
    encoder_output_dim = 128

    step1_di_output_dim = 50
    step2_rnn2_hidden_dim = 80
    attention_hidden_dim = 100

    encoder_args = {
        'part1_args': {
            'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': part1_hidden_dim, 'rnn_type': 'LSTM'
        },
        'part2_args': {
            'p_vector_dim': part1_hidden_dim, 'm_embedding_dim': 16, 'output_dim': part2_output_dim
        },
        'part3_args': {
            'p_tilde_dim': part2_output_dim, 'conv_out_channels': encoder_output_dim, 'conv_kernel_size': 3
        }
    }

    decoder_args = {
        'step1_pt_encoder_args': {
            'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'LSTM'
        },
        'step1_m_embedding_dim': 16,
        'step1_di_output_dim': step1_di_output_dim,
        'step2_rnn2_hidden_dim': step2_rnn2_hidden_dim,
        'attention_job_encoding_dim': encoder_output_dim,
        'attention_hidden_dim': attention_hidden_dim,
        'step2_rnn_type': 'LSTM'
    }

    critic_args = {
        'encoder_output_dim': encoder_output_dim,
        'conv_channels_list': [64, 32, 16],
        'conv_kernel_sizes_list': [3, 3, 3],
        'final_fc_hidden_dims': [128]
    }

    mock_proc_times = torch.randn(batch_size, num_jobs, num_machines, 1)
    mock_num_machines = torch.full((batch_size, 1), float(num_machines))

    print(f"Processing Times Shape: {mock_proc_times.shape}")
    print(f"Num Machines Scalar Shape: {mock_num_machines.shape}")

    try:
        actor = PFSPNet(encoder_args, decoder_args)
        critic = CriticNetwork(**critic_args)
        print("Actor and Critic models instantiated successfully.")
    except Exception as e:
        print(f"Error during model instantiation: {e}")

    try:
        selected_indices, log_probs, encoded_jobs = actor(
            mock_proc_times, 
            mock_num_machines,
            max_decode_len=num_jobs
        )
        print("Actor forward pass successful.")
    except Exception as e:
        print(f"Error during Actor forward pass: {e}")

    try:
        baseline_value = critic(encoded_jobs)
        print("Critic forward pass successful.")
    except Exception as e:
        print(f"Error during Critic forward pass: {e}")
        
    print(f"Actor -> Selected Indices Shape: {selected_indices.shape}")
    print(f"         Expected: ({batch_size}, {num_jobs})")
    
    print(f"Actor -> Log Probs Shape: {log_probs.shape}")
    print(f"         Expected: ({batch_size}, {num_jobs}, {num_jobs})")

    print(f"Actor -> Encoded Jobs Shape: {encoded_jobs.shape}")
    print(f"         Expected: ({batch_size}, {num_jobs}, {encoder_output_dim})")

    print(f"Critic -> Baseline Value Shape: {baseline_value.shape}")
    print(f"          Expected: ({batch_size}, 1)")
    print("-" * 30)

    assert selected_indices.shape == (batch_size, num_jobs), "Indices shape mismatch!"
    assert log_probs.shape == (batch_size, num_jobs, num_jobs), "Log probs shape mismatch!"
    assert encoded_jobs.shape == (batch_size, num_jobs, encoder_output_dim), "Encoded jobs shape mismatch!"
    assert baseline_value.shape == (batch_size, 1), "Baseline value shape mismatch!"
"""