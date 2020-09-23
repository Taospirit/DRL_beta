config = {
    'ddpg': # copy from test_ddpg
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 1,
        'update_iteration': 10,
        'target_update_freq': 10,
        'batch_size': 128,
        'hidden_dim': 32,
        'episodes': 2000,
        'max_step': 300,
        'SAVE_DIR': '/save/ddpg_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'DDPG_',
    },
    'td3':
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 1,
        'update_iteration': 10,
        'target_update_freq': 10,
        'batch_size': 128,
        'hidden_dim': 32,
        'episodes': 2000,
        'max_step': 300,
        'SAVE_DIR': '/save/td3_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'DDPG_',
    },
    'sac':
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 5,
        'update_iteration': 10,
        'target_update_freq': 20,
        'batch_size': 128,
        'hidden_dim': 256,
        'episodes': 500,
        'max_step': 300,
        'SAVE_DIR': '/save/sac_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'SAC_',
    },
    'sac_per':
    {
        'env_name': 'Pendulum-v0',
        'buffer_size': 50000,
        'actor_learn_freq': 1,
        'update_iteration': 10,
        'target_update_freq': 10,
        'batch_size': 128,
        'hidden_dim': 32,
        'episodes': 1000,
        'max_step': 300,
        'SAVE_DIR': '/save/sac_per_',
        'PKL_DIR': '/pkl/sac_',
        'LOG_DIR': '/logs', 
        'POLT_NAME': 'SAC_',
    }

}