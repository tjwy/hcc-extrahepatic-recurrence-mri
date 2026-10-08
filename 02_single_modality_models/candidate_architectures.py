"""Prespecified image architecture candidates and training hyperparameters."""

def build_candidates():
	cfg_2d = {
		"optim": "sgd", "lr": 1e-2, "head_lr": 1e-2, "bb_lr": 1e-2,
		"momentum": 0.9, "nesterov": True, "wd": 1e-4, "dropout": 0.0,
		"batch_size": 64, "epochs": 50, "patience": 50, "use_early_stopping": False,
	}
	cfg_2d_medium = cfg_2d.copy(); cfg_2d_medium["batch_size"] = 32
	cfg_2d_heavy = cfg_2d.copy(); cfg_2d_heavy["batch_size"] = 16
	cfg_2p5d = {
		"optim": "sgd", "lr": 1e-2, "head_lr": 1e-2, "bb_lr": 1e-2,
		"momentum": 0.9, "nesterov": True, "wd": 1e-4, "dropout": 0.0,
		"batch_size": 48, "epochs": 50, "patience": 50, "use_early_stopping": False,
	}
	cfg_2p5d_heavy = cfg_2p5d.copy(); cfg_2p5d_heavy["batch_size"] = 16
	cfg_3d = {
		"optim": "sgd", "lr": 1e-2, "momentum": 0.9, "nesterov": True,
		"wd": 1e-4, "dropout": 0.0, "batch_size": 8, "epochs": 50,
		"patience": 50, "use_early_stopping": False,
	}
	return [
		("2d", "resnet18", cfg_2d.copy()),
		("2d", "resnet34", cfg_2d.copy()),
		("2d", "resnet50", cfg_2d.copy()),
		("2d", "vgg16_bn", cfg_2d_heavy.copy()),
		("2d", "vgg19_bn", cfg_2d_heavy.copy()),
		("2d", "densenet121", cfg_2d_medium.copy()),
		("2d", "densenet169", cfg_2d_heavy.copy()),
		("2d", "efficientnet_b0", cfg_2d.copy()),
		("2d", "efficientnet_b1", cfg_2d_medium.copy()),
		("2d", "mobilenet_v2", cfg_2d.copy()),
		("2p5d", "resnet18", cfg_2p5d.copy()),
		("2p5d", "resnet34", cfg_2p5d.copy()),
		("2p5d", "resnet50", cfg_2p5d.copy()),
		("2p5d", "vgg16_bn", cfg_2p5d_heavy.copy()),
		("2p5d", "vgg19_bn", cfg_2p5d_heavy.copy()),
		("2p5d", "densenet121", cfg_2p5d.copy()),
		("2p5d", "densenet169", cfg_2p5d.copy()),
		("2p5d", "efficientnet_b0", cfg_2p5d.copy()),
		("2p5d", "efficientnet_b1", cfg_2p5d.copy()),
		("2p5d", "mobilenet_v2", cfg_2p5d.copy()),
		("3d", "resnet10_3d", cfg_3d.copy()),
		("3d", "resnet18_3d", cfg_3d.copy()),
		("3d", "resnet34_3d", cfg_3d.copy()),
		("3d", "resnet50_3d", cfg_3d.copy()),
		("3d", "resnet101_3d", cfg_3d.copy()),
		("3d", "resnet152_3d", cfg_3d.copy()),
		("3d", "resnet200_3d", cfg_3d.copy()),
	]
