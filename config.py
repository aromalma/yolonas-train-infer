def get_names(path):
	with open(path,'r') as f:
		t=f.read().split("\n")
	names=[x for x in filter(lambda i : bool(len(i)),t)]
	print(names)
	return names
config={

	'model':'yolo_nas_m', # model
	'CHECKPOINT_DIR' : 'checkpoints',
	'experiment_name':'test',  # name of experiment

	'dataset_params' : {
	    'data_dir':'', # root of train and test folder
	    'train_images_dir':'train/',
	    'train_labels_dir':'train/',
	    'test_images_dir':'test/',
	    'test_labels_dir':'test/',
	    'classes': get_names('/home/pavan/ksa/classes.txt') # path to classes.txt or obj.names
	},
	'dataloader_params':{
	    'batch_size':4, # adjust based on your GPU
	    'num_workers':4
	},
	'train_parms':{
		'silent_mode': True,
	    "average_best_models":True,
	    "warmup_mode": "linear_epoch_step",
	    "warmup_initial_lr": 1e-6,
	    "lr_warmup_epochs": 3,
	    "initial_lr": 5e-4,
	    "lr_mode": "cosine",
	    "cosine_final_lr_ratio": 0.1,
	    "optimizer": "Adam",
	    "optimizer_params": {"weight_decay": 0.0001},
	    "zero_weight_decay_on_bias_and_bn": True,
	    "ema": True,
	    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
	    "max_epochs": 40,
	    "mixed_precision": True,
	}
}