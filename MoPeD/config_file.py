# config = {}
# config['seed'] = [42]
# config['epochs'] = [30]
# config['batch_size'] = [64]
# config['use_stopwords'] = [True]
# config['maxlen'] = [100]
# config['ratio'] = [[70, 10, 20]]
# config['kernel_sizes'] = [[1, 2, 3, 5]]
# config['dropout'] = [0.6]
# config['user_self_attention'] = [False]
# config['n_heads'] = [8]
# config['nb_heads'] = [4]
# config['num_classes'] = [2]
# config['target_names'] = [['real', 'fake']]

# Configuration file for MoPeD with OOC dataset

config = {
    # Basic settings
    'seed': [42],
    'task': ['ooc'],
    'maxlen': [170],  # Maximum sequence length for text
    
    # Model architecture
    'num_classes': [2],  # Binary: Pristine vs OOC
    'target_names': [['Pristine', 'OOC']],
    
    # CNN settings
    'kernel_sizes': [[2, 3, 4]],  # Kernel sizes for text CNN
    
    # Training hyperparameters
    'batch_size': [32],
    'epochs': [50],
    'dropout': [0.3],
    
    # Feature dimensions (these will be overridden by command-line args if provided)
    'kl_in_features': [512],      # Latent dimension for VAE
    'fc5_in_features': [3448],    # 862 * 4 (adjust based on your combined features)
    'fc4_in_features': [768],
    'fc3_in_features': [3448],
    'fc2_in_features': [192],
    'fc1_in_features': [96],
    
    # These will be set dynamically
    # 'embedding_weights': will be added by load_bert_embeddings()
    # 'save_path': will be set in train_and_test()
}

# Note: Values are in lists because the original Weibo/Fakeddit code
# used grid search over multiple values. The process_config function
# extracts the first element from each list.