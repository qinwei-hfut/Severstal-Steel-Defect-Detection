import gc
from common_blocks.training_helper import Trainer_cv
from common_blocks.utils import plot, set_seed
from configs.train_params import *
from common_blocks.utils import load_model_unet, load_model_fpn
from segmentation_models_pytorch import Unet, FPN, PSPNet

if __name__ == '__main__':
    set_seed()
    TOTAL_FOLDS = 1
    for cur_fold in range(0, TOTAL_FOLDS):
        print('Current FOLD {}'.format(cur_fold))
        model_trainer = Trainer_cv(load_model_fpn(model_weights),
                                   num_epochs,
                                   cur_fold,
                                   batch_size=BATCH_SIZE)
        # model_trainer.start()
        # model_trainer.evaluate()
        model_trainer.evaluate_mt()

        plot(model_trainer.losses, "BCE-DICE loss", cur_fold)
        plot(model_trainer.dice_scores, "Dice score", cur_fold)
