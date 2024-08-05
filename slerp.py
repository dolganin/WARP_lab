from transformers import AutoModelForCausalLM
from config_parse import ConfigParser
import torch
def main_slerp(theta_init, theta_list, lambd) -> AutoModelForCausalLM:
    """
    Spherical linear interpolation.

    Args:
        warp: Warp object
        theta_init: initial angle
        theta_list: list of angles

    Returns:
        warp: Warp object
    """
    config = ConfigParser().config

    theta_0 = theta_init
    theta_1 = theta_list[0]

    for theta_m in theta_list[1:]:
        result = calculate_delta(theta_0.state_dict(), theta_1.state_dict(), theta_m.state_dict(), lambd)
        theta_1 = result
    return AutoModelForCausalLM.from_pretrained(config["warp_model_parameters"]["sft_model"], state_dict=result)

def calculate_delta(init, theta_a, theta_b, lambd):
    slerp_state_dict = {}
    for key in init.keys():
        theta_init_vec = init[key]

        theta_0 = theta_a[key]
        theta_1 = theta_b[key]

        delta_0 = theta_0 - theta_init_vec
        delta_1 = theta_1 - theta_init_vec

        omega = torch.arccos((torch.dot(torch.ravel(delta_0), torch.ravel(delta_1)) / (torch.norm(delta_0) * torch.norm(delta_1))))
        delta = torch.sin((1-lambd)*omega) / torch.sin(omega) * delta_0 + torch.sin(lambd*omega) / torch.sin(omega) * delta_1
        slerp_state_dict[key] = delta + init[key]
    return slerp_state_dict


