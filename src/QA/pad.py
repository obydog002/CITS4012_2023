import numpy as np

class Pad:
    def cut(desired_size, items):
        for elt in items:
            if len(elt) > desired_size:
                del elt[desired_size:]

    def pad(desired_size, items, target=False):
        # embedding size for all should be the same.
        if not target:
            emb_size = np.shape(items[0][0])[0]
        for i,_ in enumerate(items):
            elt_len = len(items[i])
            if elt_len < desired_size:
                if target: # append empty answer to target
                    items[i].extend(['OOA'] * (desired_size - elt_len))
                else: # pad empty arrays
                    items[i].extend([np.array([0] * emb_size) for x in range(elt_len, desired_size)])

    def cut_pad_to(desired_size, items, target=False):
        Pad.cut(desired_size, items)
        Pad.pad(desired_size, items, target=target)

    def get_max(items):
        return max(len(item) for item in items)

    def get_min(items):
        return min(len(item) for item in items)

    # convert unrolled string targets to integers
    def convert_targets(target_list, target2int):
        for i, targets in enumerate(target_list):
            for j, target in enumerate(targets):
                target_list[i][j] = target2int[target]
