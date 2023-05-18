import pickle

with open("results.pkl", "rb") as f:
    results = pickle.load(f)

    for i, items in enumerate(results.items()):
        key = items[0]
        value = items[1]
        d = dict(key)
        iters_inc = d["iters_inc"]
        total_inc = 0
        print(f"{i}: {value.keys()}")
        for inc in iters_inc:
            total_inc += inc
            if total_inc not in value:
                print(f"row {i}, key: {key} has missing its")
                break
