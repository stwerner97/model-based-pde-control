from abc import abstractmethod

import pytorch_lightning as pl
from munch import munchify

from pdecontrol.surrogates.common.datamodule import PDEDataModule
from pdecontrol.surrogates.training import PDETrainingModule
from pdecontrol.surrogates.surrogate import PDESurrogate


class PDESurrogateFactory:
    def surrogate(self, **kwargs):
        return PDESurrogate(**kwargs)

    @abstractmethod
    def model(self, **kwargs):
        pass

    @property
    def defaults(self):
        model = {}
        surrogate = {}
        training = {}
        trainer = {}
        curriculum = {}

        params = {
            "model": model,
            "surrogate": surrogate,
            "training": training,
            "trainer": trainer,
            "curriculum": curriculum,
        }
        return munchify(params)
