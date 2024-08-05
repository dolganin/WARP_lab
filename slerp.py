from transformers import AutoModelForCausalLM
from config_parse import ConfigParser
import torch

def main_slerp(theta_init: torch.Tensor, theta_list: list[torch.Tensor], lambd: float) -> AutoModelForCausalLM:
    """
    Performs spherical linear interpolation (SLERP) between a set of angles.

    Args:
        theta_init: The initial angle tensor.
        theta_list: A list of angle tensors for interpolation.
        lambd: The interpolation weight (between 0 and 1).

    Returns:
        An AutoModelForCausalLM instance loaded with the interpolated model state.
    """

    config = ConfigParser().config

    theta_0 = theta_init
    theta_1 = theta_list[0]

    for theta_m in theta_list[1:]:
        result = calculate_delta(theta_0, theta_1, theta_m, lambd)
        theta_1 = result

    return AutoModelForCausalLM.from_pretrained(config["warp_model_parameters"]["sft_model"], state_dict=result)

def calculate_delta(init: torch.Tensor, theta_a: torch.Tensor, theta_b: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    Calculates the delta value for a single step of spherical linear interpolation.

    Args:
        init: The initial state dictionary.
        theta_a: The first angle tensor.
        theta_b: The second angle tensor.
        lambd: The interpolation weight (between 0 and 1).

    Returns:
        A dictionary containing the delta values for each key in the initial state.
    """

    slerp_state_dict = {}
    for key in init.keys():
        theta_init_vec = init[key]

        theta_0 = theta_a[key]
        theta_1 = theta_b[key]

        delta_0 = theta_0 - theta_init_vec
        delta_1 = theta_1 - theta_init_vec

        # Calculating the angle between the two high dimensional vectors.
        omega = torch.arccos((torch.dot(torch.ravel(delta_0), torch.ravel(delta_1)) /
                             (torch.norm(delta_0) * torch.norm(delta_1))))
        delta = torch.sin((1 - lambd) * omega) / torch.sin(omega) * delta_0 + \
                torch.sin(lambd * omega) / torch.sin(omega) * delta_1
        slerp_state_dict[key] = delta + init[key]

    return slerp_state_dict
