from abstract_trainer import AbstractTrainer

class DDaCLAETrainer(AbstractTrainer):
    """Class implementing the DDaCLAE algorithm for CL training"""
    def __init__(self, config):
        self.model = config["model"]
        self.data = config["data"] 



    def train(self, x, d):
        """
        at each timestep:
        a) estimate model theta
        b) filter training data so that you only include those where b <= theta
        c) run a training pass 
        """


        for e in self.num_epochs:
            # estimate theta
            theta_hat = self.estimate_theta() 

            # filter out needed training examples from full set
            epoch_idx = self.filter_examples(theta_hat) 

            # create a new pytorch dataset/dataloader
            epoch_training_data = self.build_epoch_training_set(epoch_idx) 

            # run one training step 
            # calculate loss and backprop 

            # training 
            self.model.train() 
            for j, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch) 
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3]
                        }
                outputs = self.model(**inputs) 
                loss = outputs[0]
                loss.backward() 
                self.optimizer.step() 
                self.scheduler.step()
                self.model.zero_grad()

            # eval 
            self.model.eval()


            # save model to disk if it's best performing so far 
            self.model.save() 


    def estimate_theta(self):
        pass 
        

    def filter_examples(self, theta_hat):
        pass 


    def build_epoch_training_set(self, epoch_idx):
        pass 




