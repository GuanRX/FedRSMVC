# server/FedAvg_server.py
import copy
import torch


def aggregate(client_updates_list):
    if not client_updates_list:
        return None

    num_clients = len(client_updates_list)
    averaged_params = copy.deepcopy(client_updates_list[0])
    if num_clients == 1:
        return averaged_params

    for key in averaged_params.keys():
        if isinstance(averaged_params[key], torch.Tensor):
            target_device = averaged_params[key].device
            for i in range(1, num_clients):
                client_param = client_updates_list[i][key]
                if client_param.device != target_device:
                    client_param = client_param.to(target_device)

                averaged_params[key] += client_param
            averaged_params[key] = torch.div(averaged_params[key], num_clients)

    return averaged_params