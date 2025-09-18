import torch
import os
from actor_critic import PFSPNet
from calc import calculate_objectives_pytorch

def run_inference(model_path, instance_path, num_samples=100, device='cpu'):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        return
    if not os.path.exists(instance_path):
        print(f"Error: Instance file not found {instance_path}")
        return

    print(f"Starting inference...")
    print(f"Loading instance: {instance_path}")
    instance_data = torch.load(instance_path, map_location=device)
    P_instance = instance_data['P_instance'].to(device)
    E_instance = instance_data['E_instance'].to(device)
    R_instance = instance_data['R_instance'].to(device)
    u_starts = instance_data['u_starts'].to(device)
    s_durations = instance_data['s_durations'].to(device)
    f_factors = instance_data['f_factors'].to(device)
    
    config_num_jobs = P_instance.shape[0]
    config_num_machines = P_instance.shape[1]

    print("Building model architecture...")
    enc_part1_args = {'scalar_input_dim': 1, 'embedding_dim': 32, 'hidden_dim': 64, 'rnn_type': 'RNN', 'num_rnn_layers':1}
    enc_part2_args = {'p_vector_dim': enc_part1_args['hidden_dim'], 'm_embedding_dim': 16, 'output_dim': 48}
    ENC_OUT_CHANNELS = 96
    enc_part3_args = {'p_tilde_dim': enc_part2_args['output_dim'], 'conv_out_channels': ENC_OUT_CHANNELS, 'conv_kernel_size': 3, 'conv_padding': 'same'}
    encoder_config_args = {'part1_args': enc_part1_args, 'part2_args': enc_part2_args, 'part3_args': enc_part3_args}
    dec_step1_pt_enc_args = {'scalar_input_dim': 1, 'embedding_dim': 30, 'hidden_dim': 50, 'rnn_type': 'RNN', 'num_rnn_layers': 1}
    decoder_config_args = {
        'step1_pt_encoder_args': dec_step1_pt_enc_args,
        'step1_m_embedding_dim': 20, 'step1_di_output_dim': 40, 'step1_fc_hidden_dims': [35],
        'step2_rnn2_hidden_dim': 70, 'step2_rnn_type': 'LSTM', 'step2_num_rnn_layers': 1,
        'attention_job_encoding_dim': ENC_OUT_CHANNELS,
        'attention_hidden_dim': 55,
        'ptr_dim': 25
    }
    
    actor_model = PFSPNet(encoder_args=encoder_config_args, decoder_args=decoder_config_args).to(device)
    
    print(f"Loading model weights: {model_path}")
    actor_model.load_state_dict(torch.load(model_path, map_location=device))
    
    actor_model.eval()
    
    instance_features = P_instance.unsqueeze(-1)
    batch_features = instance_features.unsqueeze(0).expand(num_samples, -1, -1, -1)
    dummy_m_scalar = torch.full((num_samples, 1), float(config_num_machines), device=device)

    print(f"Model is generating {num_samples} candidate solutions...")
    with torch.no_grad():
        candidate_sequences, _, _ = actor_model(
            batch_features, dummy_m_scalar, max_decode_len=config_num_jobs
        )
        
        put_off_eval = torch.zeros(num_samples, config_num_machines, config_num_jobs, device=device, dtype=torch.long)
        cmax_values, _ = calculate_objectives_pytorch(
            job_sequences=candidate_sequences,
            put_off_matrices=put_off_eval,
            P=P_instance, E=E_instance, R=R_instance, u=u_starts, s=s_durations, f=f_factors,
            device=device
        )
        
        best_cmax, best_idx = torch.min(cmax_values, dim=0)
        best_sequence = candidate_sequences[best_idx]

    print("\nInference completed...")
    print(f"The best sequence of {num_samples} samples is:")
    print(best_sequence.tolist())
    print(f"\nThe best Cmax value is: {best_cmax.item():.2f}")

if __name__ == '__main__':
    MODEL_FILE_PATH = 'PFSPNET/models/best_actor_10j_3m.pt'
    INSTANCE_FILE_PATH = 'PFSPNET/data/instance_10j_3m_10k.pt'
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_inference(
        model_path=MODEL_FILE_PATH,
        instance_path=INSTANCE_FILE_PATH,
        num_samples=128,
        device=DEVICE
    )
