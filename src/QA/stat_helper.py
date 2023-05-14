class StatHelper:
    def get_class_weights(target_list, classes):
        freqs = [0]*classes
        for targets in target_list:
            for target in targets:
                freqs[target] += 1
        biggest_class = max(freqs)
        return [biggest_class/x for x in freqs]

