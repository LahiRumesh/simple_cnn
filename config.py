from easydict import EasyDict


cfg = EasyDict()

cfg.data_dir = 'data_dir' # Image Folder path which contain train and val folders
cfg.device = '0' # cuda device, i.e. 0 or 0,1,2,3 

cfg.image_size = 224 #input image size
cfg.batch_size = 4 # batch size
cfg.epochs = 40 #number of epochs
cfg.learning_rate = 0.001
cfg.momentum = 0.9
cfg.weight_decay = 0


#model configuration
cfg.model = 'resnet50' # model architecture
cfg.pretrained = True # Use pretrained weight
cfg.loss_criterion = 'CrossEntropyLoss' # Loss Function
cfg.optimizer = 'SGD' # optimizer
cfg.lr_scheduler = 'StepLR' # Learning Rate Decay 
cfg.steps = 7 # Decay epochs 
cfg.gamma = 0.1  # Decay Learning Rate factor 

#Early Stopping
cfg.use_early_stopping = True #Use Early stopping
cfg.patience = 8 # how many epochs to wait before stopping when accuracy is not improving
cfg.min_delta = 0.0001 # minimum difference between new accuracy and old accuracy for new accuracy to be considered as an improvement