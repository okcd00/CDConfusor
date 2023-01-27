"""
@Time   :   2021-01-21 11:10:52
@File   :   bases.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import logging
import pytorch_lightning as pl

from bbcm.solver.build import make_optimizer, build_lr_scheduler


class BaseTrainingEngine(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._logger = logging.getLogger(cfg.MODEL.NAME)

    def configure_optimizers(self):
        show_logs = False
        if self.trainer.is_global_zero:
            show_logs = True
        optimizer = make_optimizer(self.cfg, self, debug=show_logs)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [scheduler]
    
    def on_validation_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self._logger.info('\n===============\n*    Valid    *\n===============\n')

    def on_test_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self._logger.info('\n===============\n*   Testing   *\n===============\n')
