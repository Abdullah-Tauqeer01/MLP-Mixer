def get_mixer_config(name, hidden_dim, num_blocks,  channels_mlp_dim,tokens_mlp_dim, patch_size, sequence_length):
    return {
        'name': name,
        'patches': {'size': patch_size},
        'hidden_dim': hidden_dim,
        'num_blocks': num_blocks,
        'channels_mlp_dim': channels_mlp_dim,
        'tokens_mlp_dim': tokens_mlp_dim,
        'sequence_length': sequence_length
    }

CONFIGS = {
    'S/32': get_mixer_config('S/32', 512, 8, 2048, 256, (32, 32), 49),
    'S/16': get_mixer_config('S/16', 512, 8, 2048, 256, (16, 16), 196),
    'B/32': get_mixer_config('B/32', 768, 12, 3072, 384, (32, 32), 49),
    'B/16': get_mixer_config('B/16', 768, 12, 3072, 384, (16, 16), 196),
    'L/32': get_mixer_config('L/32', 1024, 24, 4096, 512, (32, 32), 49),
    'L/16': get_mixer_config('L/16', 1024, 24, 4096, 512, (16, 16), 196),
    'H/14': get_mixer_config('H/14', 1280, 32, 5120, 640, (14, 14), 256)
}



    