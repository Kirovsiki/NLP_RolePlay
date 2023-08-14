from parlai.scripts.train_model import TrainModel

def main():
    # set up the parameters
    params = {
        'task': 'empathetic_dialogues,wizard_of_wikipedia,guguaitrain,cornell_movie',
        'model': 'transformer/generator',
        'model_file': 'model/guguAI19',
        'init_model': 'data/models/blender/blender_90M/model',
        'dict_file': 'data/models/blender/blender_90M/model.dict',
        'fp16': True,
        'embedding_size': 512,
        'ffn_size': 2048,
        'n_positions': 512,
        'dropout': 0.1,
        'attention_dropout': 0.0,
        'n_layers': 8,
        'n_heads': 16,
        'learn_positional_embeddings': True,
        'variant': 'xlm',
        'num_epochs': 200,
        'validation_every_n_secs': 3600,
        'batchsize': 50,
        'activation': 'gelu',
        'optimizer': 'adamax',
        'lr_scheduler': 'fixed',
        'gradient_clip': 0.1,
        'dict_tokenizer': 'bpe',
        'dict_lower': True,
        'lr': 8e-04,
        'text_truncate': 512,
        'label_truncate': 128,
        'save_after_valid': True,
        'gpu': 0,  # add this line to use GPU
        'dict_maxtokens': 100000  # limit the vocabulary size to 20000
    }

    # create the TrainModel object and start training
    TrainModel.main(**params)

if __name__ == '__main__':
    main()
