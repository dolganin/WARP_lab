import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import copy
from torch.optim import AdamW
from torch.nn.functional import log_softmax
from slerp import main_slerp
from os import path


class WarpTrainer:
    """
    A class used to train a model using the WARP (Weighted Averaging and Regularized Policy) algorithm.
    
    Attributes
    ----------
    config : dict
        Configuration parameters for the training.
    dataloader : DataLoader
        DataLoader for the training data.
    device : torch.device
        Device to perform the training on (CPU or GPU).
    sft : AutoModelForCausalLM
        Model used for causal language modeling.
    sft_tokenizer : AutoTokenizer
        Tokenizer for the sft model.
    reward_tokenizer : AutoTokenizer
        Tokenizer for the reward model.
    reward_model : AutoModelForSequenceClassification
        Model used to calculate reward.
    """
    
    def __init__(self, config, dataloader):
        """
        Initializes the WarpTrainer with the given configuration and dataloader.
        
        Parameters
        ----------
        config : dict
            Configuration parameters for the training.
        dataloader : DataLoader
            DataLoader for the training data.
        """
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sft = AutoModelForCausalLM.from_pretrained(self.config["warp_model_parameters"]["sft_model"])
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.config["warp_model_parameters"]["reward_model"], num_labels=1).to(self.device)
        
        self.sft_tokenizer = AutoTokenizer.from_pretrained(self.config["warp_model_parameters"]["sft_model"], use_fast=True, padding_side="left")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.config["warp_model_parameters"]["reward_model"], use_fast=True)

        self.sft_tokenizer.pad_token = self.sft_tokenizer.eos_token

        self.dataloader = dataloader

    def train(self):
        """
        Trains the model using the WARP algorithm.
        
        The training consists of multiple iterations where model parameters are updated using policy gradients
        and KL-regularized rewards. The weights are averaged using SLERP (Spherical Linear Interpolation).
        """
        init_theta = copy.deepcopy(self.sft)
        for i in range(self.config["warp_model_parameters"]["iterations"]):
            theta_list = []
            for j in range(self.config["warp_model_parameters"]["m"]):
                theta = copy.deepcopy(init_theta).to(self.device)
                theta_ema = copy.deepcopy(init_theta).to(self.device)
                optimizer = AdamW(theta.parameters(), lr=self.config["warp_model_parameters"]["learning_rate"])
                for iteration, batch in enumerate(self.dataloader):
                    if iteration > self.config["warp_model_parameters"]["t"]:
                        break
                    
                    optimizer.zero_grad()

                    theta_ends, theta_probs = self.generate_end(theta, self.sft_tokenizer, batch)
                    theta_ema_ends, sft_probs = self.generate_end(theta_ema, self.sft_tokenizer, batch)

                    reward_1 = self.get_reward(theta_ends)
                    reward_2 = self.get_reward(theta_ema_ends)

                    reward = (reward_1 + reward_2)/2



                    kl_reward = self.kl_regularize(sft_probs, reward, theta_probs)
                    policy_grad = self.get_policy_gradient(theta_probs, kl_reward)

                    policy_grad.backward()

                    optimizer.step()

                    print("Iteration: {}, reward: {}, loss: {}".format(iteration, kl_reward, policy_grad))

                theta_list.append(theta.to("cpu"))
            theta_slerp = main_slerp(init_theta, theta_list, lambd=self.config["warp_model_parameters"]["lambda"])
            init_theta = self.weights_averaging(init_theta, theta_slerp, self.config["warp_model_parameters"]["alpha"])
            
        self.sft = self.weights_averaging(self.sft, theta_slerp, self.config["warp_model_parameters"]["alpha"])
    
    def generate_end(self, model, tokenizer, batch):
        """
        Generates the end sequence and log probabilities for the given batch.
        
        parameters:
        model : PreTrainedModel
            The model to generate the sequence.
        tokenizer : PreTrainedTokenizer
            The tokenizer for the model.
        batch : list
            The input batch for generation.
        
        returns:
        list
            Decoded output sequences.
        torch.Tensor
            Log probabilities of the generated sequences.
        """
        token_input = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(self.device)
        output = model.generate(**token_input, max_length=self.config["warp_model_parameters"]["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)
        decoded_output = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
        log_probs = log_softmax(model(output).logits, dim=-1)
        return decoded_output, log_probs

    def weights_averaging(self, model_1, model_2, alpha) -> AutoModelForCausalLM:
        """
        Averages the weights of two models with a given alpha.
        
        parameters:
        model_1 : PreTrainedModel
            The first model.
        model_2 : PreTrainedModel
            The second model.
        alpha : float
            The weighting factor for averaging.
        
        returns:
        PreTrainedModel
            The model with averaged weights.
        """
        averaged_model_dict = {}
        for key in model_1.state_dict().keys():
            averaged_model_dict[key] = (1 - alpha) * model_1.state_dict()[key] + alpha * model_2.state_dict()[key]

        averaged_model = AutoModelForCausalLM.from_pretrained(self.config["warp_model_parameters"]["sft_model"],
                                                              state_dict=averaged_model_dict)
        return averaged_model
    
    def get_reward(self, batch):
        """
        Calculates the reward for a given batch.
        
        parameters:
        batch : list
            The input batch.
        
        returns
        torch.Tensor
            The calculated reward.
        """
        model_input = self.reward_tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        return torch.sigmoid(self.reward_model(**model_input).logits)
    
    def kl_regularize(self, log_probs, reward, alignment_probs):
        """
        Applies KL regularization to the reward.
        
        parameters
        log_probs : torch.Tensor
            Log probabilities of the generated sequences.
        reward : torch.Tensor
            The calculated reward.
        alignment_probs : torch.Tensor
            Alignment probabilities.
        
        returns
        torch.Tensor
            KL-regularized reward.
        """
        try:
            kl_div = torch.mean(alignment_probs - log_probs)
        except:
            kl_div = torch.mean(alignment_probs) - torch.mean(alignment_probs)

        return torch.mean(reward - self.config["warp_model_parameters"]["beta"] * kl_div)
    
    def get_policy_gradient(self, log_probs, kl_reward):
        """
        Calculates the policy gradient.
        
        parameters:
        log_probs : torch.Tensor
            Log probabilities of the generated sequences.
        kl_reward : torch.Tensor
            KL-regularized reward.
        mode : str, optional
            The mode for calculating the policy gradient, by default "mean".
        
        returns:
        torch.Tensor
            The calculated policy gradient.
        """
        return -torch.mean(log_probs * kl_reward)/len(log_probs)
        
    def save(self, out_name) -> None:
        """
        Saves the trained model and tokenizer.
        
        parameters
        out_name : str
            The name for the saved output directory.
        """
        output_dir = path.join(self.config["general"]["output_dir"], out_name)
        self.sft_tokenizer.save_pretrained(output_dir+"_token")
        self.sft.save_pretrained(output_dir, from_pt=True)

    def get_model_token(self):
        return self.sft, self.sft_tokenizer

    def test(self, model, tokenizer, data):
        with torch.no_grad():
            output, probs = self.generate_end(model, tokenizer, data)
        with torch.no_grad():
            reward = self.get_reward(output)

        mean_reward = torch.mean(reward)
        return probs, mean_reward
    
        

        

