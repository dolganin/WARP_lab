from train_warp import WarpTrainer
from imdb_dataset_warp import IMDBDataset
from config_parse import ConfigParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader



def main():
    config = ConfigParser().config
    _, test_dataset = IMDBDataset(config).get_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    alligned_model = AutoModelForCausalLM.from_pretrained(config["test_allignment"]["llm"]).to(device)
    sft_model = AutoModelForCausalLM.from_pretrained(config["warp_model_parameters"]["sft_model"]).to(device)
    test_loader = DataLoader(test_dataset, batch_size=config["warp_model_parameters"]["batch_size"], shuffle=True, num_workers=15)

    tester = WarpTrainer(config, dataloader=None)
    alligned_token = AutoTokenizer.from_pretrained(config["test_allignment"]["token"], use_fast=True)
    sft_token = AutoTokenizer.from_pretrained(config["warp_model_parameters"]["sft_model"], use_fast=True, padding_side="left")

    sft_token.pad_token = sft_token.eos_token

    kl_divs = []
    alligned_rewards = []
    sft_rewards = []

    for batch in test_loader:
        alligned_probs, alligned_mean_reward = tester.test(model=alligned_model, tokenizer=alligned_token, data=batch)
        sft_probs, sft_mean_reward = tester.test(model=sft_model, tokenizer=sft_token, data=batch)
        alligned_rewards.append(alligned_mean_reward)
        sft_rewards.append(sft_mean_reward)


        kl_divs.append(kl_div(alligned_probs, sft_probs, log_target=True, reduction="batchmean"))
    
    alligned_reward = torch.mean(torch.tensor(alligned_rewards))
    sft_reward = torch.mean(torch.tensor(sft_rewards))
    mean_kl = torch.mean(torch.tensor(kl_divs))

    print(f"Reward for alligned model: {alligned_reward}")
    print(f"Reward for sft model: {sft_reward}")
    print(f"KL divergence: {mean_kl}")


if __name__ == "__main__":
    main()



        





