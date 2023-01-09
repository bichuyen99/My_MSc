from metric import BaseMetric

from utils import MetaParent, create_logger
from utils import GLOBAL_TENSORBOARD_WRITER, DEVICE

import os
import torch
from pathlib import Path
from collections import Counter

logger = create_logger(name=__name__)


class BaseCallback(metaclass=MetaParent):

    def __init__(self, model, dataloader, optimizer):
        self._model = model
        self._dataloader = dataloader
        self._optimizer = optimizer

    def __call__(self, inputs, step_num):
        raise NotImplementedError


class MetricCallback(BaseCallback, config_name='metric'):

    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            on_step,
            metrics,
            loss_prefix
    ):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._loss_prefix = loss_prefix
        self._metrics = metrics if metrics is not None else {}

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            dataloader=kwargs['dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            metrics=config.get('metrics', None),
            loss_prefix=config['loss_prefix']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            for metric_name, metric_function in self._metrics.items():
                metric_value = metric_function(
                    ground_truth=inputs[self._model.schema['ground_truth_prefix']],
                    predictions=inputs[self._model.schema['predictions_prefix']]
                )
                GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    'train/{}'.format(metric_name),
                    metric_value,
                    step_num
                )

            GLOBAL_TENSORBOARD_WRITER.add_scalar(
                'train/{}'.format(self._loss_prefix),
                inputs[self._loss_prefix],
                step_num
            )


class CheckpointCallback(BaseCallback, config_name='checkpoint'):

    def __init__(self, model, dataloader, optimizer, on_step, save_path, model_name):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._save_path = Path(os.path.join(save_path, model_name))
        if self._save_path.exists():
            logger.warning('Checkpoint path `{}` is already exists!'.format(self._save_path))
        else:
            self._save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            dataloader=kwargs['dataloader'],
            optimizer=kwargs['optimizer'],
            on_step=config['on_step'],
            save_path=config['save_path'],
            model_name=config['model_name']
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:
            logger.debug('Saving model state on step {}...'.format(step_num))
            torch.save(
                {
                    'step_num': step_num,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                },
                os.path.join(self._save_path, 'checkpoint_{}.pt'.format(step_num))
            )
            logger.debug('Saving done!')


class QualityCheckCallbackCheck(BaseCallback, config_name='validation'):
    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            loss_function,
            on_step,
            metrics=None
    ):
        super().__init__(model, dataloader, optimizer)
        self._on_step = on_step
        self._metrics = metrics if metrics is not None else {}
        self._loss_function = loss_function

    @classmethod
    def create_from_config(cls, config, **kwargs):
        metrics = {
            metric_name: BaseMetric.create_from_config(metric_cfg, **kwargs)
            for metric_name, metric_cfg in config['metrics'].items()
        }

        return cls(
            model=kwargs['model'],
            dataloader=kwargs['dataloader'],
            optimizer=kwargs['optimizer'],
            loss_function=kwargs['loss_function'],
            on_step=config['on_step'],
            metrics=metrics,
        )

    def __call__(self, inputs, step_num):
        if step_num % self._on_step == 0:  # TODO Add time monitoring
            logger.debug('Validation on step {}...'.format(step_num))
            running_params = Counter()

            self._model.eval()
            with torch.no_grad():
                for batch in self._dataloader:

                    for key, value in batch.items():
                        batch[key] = value.to(DEVICE)

                    predicts = self._model(batch).squeeze()
                    labels = batch['labels'].float()
                    loss = self._loss_function(predicts, labels)

                    for metric_name, metric_function in self._metrics.items():
                        running_params[metric_name] += metric_function(
                            inputs={'predicts': predicts.cpu(), 'labels': labels.cpu()}
                        )

                    running_params['loss'] += loss

            for label, value in running_params.items():
                GLOBAL_TENSORBOARD_WRITER.add_scalar(
                    'validation/{}'.format(label),
                    value / len(self._dataloader),
                    step_num
                )

            logger.debug('Validation on step {} is done!'.format(step_num))


class CompositeCallback(BaseCallback, config_name='composite'):
    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            callbacks
    ):
        super().__init__(model, dataloader, optimizer)
        self._callbacks = callbacks

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            model=kwargs['model'],
            dataloader=kwargs['dataloader'],
            optimizer=kwargs['optimizer'],
            callbacks=[
                BaseCallback.create_from_config(cfg, **kwargs)
                for cfg in config['callbacks']
            ]
        )

    def __call__(self, inputs, step_num):
        for callback in self._callbacks:
            callback(inputs, step_num)
