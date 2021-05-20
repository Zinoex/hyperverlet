from torch.nn import L1Loss

from hyperverlet.loss import TimeDecayMSELoss, MeanNormLoss


def construct_loss(train_args):
    criterion = train_args['criterion']

    if train_args == 'TimeDecayMSELoss':
        time_decay = train_args["time_decay"]
        return TimeDecayMSELoss(time_decay)
    else:
        losses = {
            'MeanNormLoss': MeanNormLoss,
            'L1Loss': L1Loss
        }

        return losses[criterion]()
