import matplotlib.pyplot as plt

# Helper functions for various operations on the results.pkl files
class ResultsHelper:
    # catersian product of list of dicts
    def cart(list_dict1, list_dict2):
        l = []
        for d1 in list_dict1:
            for d2 in list_dict2:
                l.append(d1 | d2)
        return l

    # returns all matching params specified by presence of fields in param
    def get_matching_params(param, all_params):
        matched = []
        for candidate in all_params:
            if type(candidate) == frozenset:
                candidate = dict(candidate)
            if param.items() <= candidate.items():
                matched.append(candidate)

        return matched

    # assumes models is a dict of frozenset:models
    # returns a dict in the same form as models
    def get_matching_models(param, models):
        matched = {}
        for key, value in models.items():
            if type(key) != frozenset:
                assert False # sanity check keys are all frozen sets
            unfrozen_key = dict(key)
            if param.items() <= unfrozen_key.items():
                matched[key] = value

        return matched

    # plot precisions, recalls, f scores for the given params
    # on the given iteration
    def plot_metrics_for_one_it(params, var_X=None, var_X_pretty_name=None, it=1, train=True):
        precisions = []
        recalls = []
        f1s = []
        x_labels = []
        
        x_labels = range(len(params))
        if var_X != None:
            x_labels = []

        metric_string = "train_report"
        if not train:
            metric_string = "test_report"

        for key, value in params.items():
            model_results = value[it][metric_string]["1"]
            precisions.append(model_results["precision"])
            recalls.append(model_results["recall"])
            f1s.append(model_results["f1-score"])
            if var_X != None:
                x_labels.append(str(dict(key)[var_X]))

        plt.plot(x_labels, precisions, label="Precision")
        plt.plot(x_labels, recalls, label="Recall")
        plt.plot(x_labels, f1s, label="F1")
        if var_X != None:
            if var_X_pretty_name != None:
                plt.xlabel(var_X_pretty_name)
            else:
                plt.xlabel(var_X)

    # plot the mtrics for one param, over all its iterations
    def plot_metrics_over_its(param, train=True):
        precisions = []
        recalls = []
        f1s = []
        its = []

        metric_string = "train_report"
        if not train:
            metric_string = "test_report"
        
        # should only be 1 value
        for value in param.values():
            for it in value.keys():
                its.append(it)
                model_results = value[it][metric_string]["1"]
                precisions.append(model_results["precision"])
                recalls.append(model_results["recall"])
                f1s.append(model_results["f1-score"])

        plt.plot(its, precisions, label="Precision")
        plt.plot(its, recalls, label="Recall")
        plt.plot(its, f1s, label="F1")


