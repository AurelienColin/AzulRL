import numpy as np


def to_hot_encoded(choices, *vmaxs):
    subarrays = [[] for _ in range(len(vmaxs))]

    for i, vmax in enumerate(vmaxs):
        for j, player_choices in enumerate(choices):
            subarrays[i].append([])
            for turn_choice in player_choices:
                hot_encoded = np.zeros(vmax)
                if turn_choice[i] != -1:
                    hot_encoded[turn_choice[i]] = 1
                subarrays[i][j].append(hot_encoded)
            subarrays[i][j] = np.array(subarrays[i][j])
    return subarrays
