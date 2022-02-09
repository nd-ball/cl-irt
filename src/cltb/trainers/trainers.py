from trainers.cbcl_trainer import CBCLTrainer
from trainers.crossreview_trainer import CrossreviewTrainer
from trainers.ddaclae_trainer import DDaCLAETrainer
from trainers.graves_trainer import GravesTrainer
from trainers.hacohen_trainer import HacohenTrainer
from trainers.mentornet_trainer import MentornetTrainer
from trainers.rbf_trainer import RbFTrainer


TRAINERS = {
    "CBCL": CBCLTrainer,
    "Crossreview": CrossreviewTrainer,
    "DDaCLAE": DDaCLAETrainer,
    "Graves": GravesTrainer,
    "Hacohen": HacohenTrainer,
    "Mentornet": MentornetTrainer,
    "RbF": RbFTrainer
}
