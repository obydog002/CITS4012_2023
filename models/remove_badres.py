import pickle

results = pickle.load(open("results.pkl", "rb"))
deletes = []
for i in results.keys():
    d = set(i)
    if ("attention_type", "Tanh") in d and ("hidden_type", "GRU") not in d:
        deletes.append(i)

del results[deletes[0]]
del results[deletes[1]]

print(results.keys())

pickle.dump(results, open("results.pkl", "wb"))

