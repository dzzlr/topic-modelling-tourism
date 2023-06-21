import random
from itertools import chain

rawdocs = [
    # "eat turkey on turkey day holiday",
    # "i like to eat cake on holiday",
    # "turkey trot race on thanksgiving holiday",
    # "snail race the turtle",
    # "time travel space race",
    # "movie on thanksgiving",
    # "movie at air and space museum is cool movie",
    # "aspiring movie star"
    'lokasi curug indah sejuk baik kondisi jalan sempit becek hujan',
    'saran hujan jalan curug licin becek ojek antar curug',
    'tempat bagus akses curug jalan tanah hujan anjur',
]

docs = [doc.split() for doc in rawdocs]
vocab = list(dict.fromkeys(chain.from_iterable(docs)))

# for i in range(len(docs)):
#     docs[i] = [vocab.index(word) for word in docs[i]]

K = 2

wt = [[0] * len(vocab) for _ in range(K)]
ta = {f"doc{i+1}": [0] * len(doc) for i, doc in enumerate(docs)}
dt = [[0] * K for _ in range(len(docs))]

random.seed(1234)
for d, doc in enumerate(docs):
    for w, word in enumerate(doc):
        ta[f"doc{d+1}"][w] = random.choice(range(1, K+1))
        ti = ta[f"doc{d+1}"][w]
        wi = vocab.index(word)
        wt[ti-1][wi] += 1

    for t in range(K):
        dt[d][t] = ta[f"doc{d+1}"].count(t+1)

alpha = 1
eta = 1

for d, doc in enumerate(docs):
    # print(docs)
    for w, word in enumerate(doc):
        t0 = ta[f"doc{d+1}"][w]
        wid = vocab.index(word)

        dt[d][t0-1] -= 1
        wt[t0-1][wid] -= 1

        left = [(wt[k][wid] + eta) / (sum(wt[k]) + len(vocab) * eta) for k in range(K)]
        right = [(dt[d][t] + alpha) / (sum(dt[d]) + K * alpha) for t in range(K)]

        probs = [l * r for l, r in zip(left, right)]
        # print(probs)
        t1 = random.choices(range(1, K+1), weights=probs)[0]

        ta[f"doc{d+1}"][w] = t1
        dt[d][t1-1] += 1
        wt[t1-1][wid] += 1


print(ta)
print(wt)
print(dt)
# wt = [
#     [0,3,1,1,1,1,1,0,2,3,1,0,1,1,1,0,1,1,1],
#     [1,1,0,0,0,0,2,1,0,0,0,1,0,0,0,1,0,0,0]
# ]
print(sum(wt[0]), sum(wt[1]))

# dt = [[8,2],[6,3],[6,2]]

for d, doc in enumerate(docs):
    # print(docs)
    for w, word in enumerate(doc):
        t0 = ta[f"doc{d+1}"][w]
        wid = vocab.index(word)

        left = [(wt[k][wid] + eta) / (sum(wt[k]) + len(vocab) * eta) for k in range(K)]
        right = [(dt[d][t] + alpha) / (sum(dt[d]) + K * alpha) for t in range(K)]

        probs = [round(l * r, 3) for l, r in zip(left, right)]
        print(probs)