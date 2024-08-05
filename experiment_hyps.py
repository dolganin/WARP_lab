from warp import WarpTrainer
from imdb_dataset_warp import IMDBDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from config_parse import ConfigParser
from torch.utils.data import DataLoader
from torch.nn.functional import kl_div
import torch

def main():
    base_config = ConfigParser().config
    exp_1_config = ConfigParser("experiments/exp1.yaml").config
    exp_2_config = ConfigParser("experiments/exp2.yaml").config
    configs = [base_config, exp_1_config, exp_2_config]

    train_dataset, test_dataset = IMDBDataset(config=base_config).get_data()
    train_loader = DataLoader(train_dataset, batch_size=base_config["warp_model_parameters"]["batch_size"], shuffle=True, drop_last=True, num_workers=15)
    test_loader = DataLoader(test_dataset, batch_size=base_config["warp_model_parameters"]["batch_size"], shuffle=False, drop_last=True, num_workers=15)

    sft_model = AutoModelForCausalLM.from_pretrained(base_config["warp_model_parameters"]["sft_model"])
    sft_tokenizer = AutoTokenizer.from_pretrained(base_config["warp_model_parameters"]["sft_model"], use_fast=True, padding_side="left")
    sft_tokenizer.pad_token = sft_tokenizer.eos_token

    exp_align_rewards = []
    exp_sft_rewards = []
    exp_kl_divs = []

    for config in configs:
        exp_trainer = WarpTrainer(config = config, dataloader=train_loader)
        exp_trainer.train()
        exp_model, exp_token = exp_trainer.get_model_token()

        kl_divs = []
        alligned_rewards = []
        sft_rewards = []

        exp_model = exp_model.to(exp_trainer.device)
        sft_model = sft_model.to(exp_trainer.device)
        for batch in test_loader:
            
            alligned_probs, alligned_mean_reward = exp_trainer.test(model=exp_model, tokenizer=exp_token, data=batch)
            sft_probs, sft_mean_reward = exp_trainer.test(model=sft_model, tokenizer=sft_tokenizer, data=batch)
            alligned_rewards.append(alligned_mean_reward)
            sft_rewards.append(sft_mean_reward)
            try:
                kl_divs.append(kl_div(alligned_probs, sft_probs, log_target=True, reduction="batchmean"))
            except:
                alligned_probs_shape = alligned_probs.shape
                kl_divs.append(kl_div(alligned_probs, sft_probs[:alligned_probs_shape[1]], log_target=True, reduction="batchmean"))
            finally:
                continue

        
        exp_kl_divs.append(torch.mean(torch.tensor(kl_divs)).item())
        exp_align_rewards.append(torch.mean(torch.tensor(alligned_rewards)).item())
        exp_sft_rewards.append(torch.mean(torch.tensor(sft_rewards)).item())
    
    print(f"Aligned model rewards: {exp_align_rewards}")
    print(f"SFT model rewards: {exp_sft_rewards}")
    print(f"KL-divergence: {exp_kl_divs}")

if __name__ == "__main__":
    main()


        