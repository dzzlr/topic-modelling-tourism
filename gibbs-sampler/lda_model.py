import random
from itertools import chain

def LDA1(docs, vocab, K, alpha, eta, iterations):
    # initialize count matrices
    # @wt : word-topic matrix
    wt = [[0] * len(vocab) for _ in range(K)]
    wt = {i: row for i, row in enumerate(wt)}
    
    # @ta : topic assignment list
    ta = {f"doc{i+1}": [0] * len(doc) for i, doc in enumerate(docs)}
    
    # @dt : counts correspond to the number of words assigned to each topic for each document
    dt = [[0] * K for _ in range(len(docs))]

    # print(len(wt[0]))
    # print(ta)
    # print(dt)
    random.seed(1234)
    for d, doc in enumerate(docs):
        # randomly assign topic to word w
        for w, word in enumerate(doc):
            ta[f"doc{d+1}"][w] = random.choice(range(1, K+1))
            
            # extract the topic index, word id and update the corresponding cell
            # in the word-topic count matrix
            ti = ta[f"doc{d+1}"][w]
            wi = vocab.index(word)
            # print(ti, wi)
            wt[ti-1][wi] += 1

        # count words in document d assigned to each topic t
        for t in range(K):
            dt[d][t] = ta[f"doc{d+1}"].count(t+1)
    
    # print('docs:', docs)
    # print('vocab:', vocab)
    print('ta:', ta)
    print('wt:', wt)
    print('dt:', dt)
    
    # for each pass through the corpus
    for _ in range(iterations):
        # for each document
        for d, doc in enumerate(docs):
            print(docs)
            # for each word
            for w, word in enumerate(doc):
                t0 = ta[f"doc{d+1}"][w]
                # print(word)
                wid = vocab.index(word)

                # print(f'd: {d}, t0: {t0}, wid: {wid}')
                
                dt[d][t0-1] -= 1
                wt[t0-1][wid] -= 1
                
                left = [(wt[k][wid] + eta) / (sum(wt[k]) + len(vocab) * eta) for k in range(K)]
                right = [(dt[d][t] + alpha) / (sum(dt[d]) + K * alpha) for t in range(K)]
                
                probs = [l * r for l, r in zip(left, right)]
                # print(probs)
                t1 = random.choices(range(1, K+1), weights=probs)[0]
                
                # update topic assignment list with newly sampled topic for token w.
                # and re-increment word-topic and document-topic count matrices with
                # the new sampled topic for token w.
                ta[f"doc{d+1}"][w] = t1
                dt[d][t1-1] += 1
                wt[t1-1][wid] += 1
                
                # examine when topic assignments change
                # if t0 != t1:
                #     print(f"doc:{d+1} token:{w} topic:{t0}=>{t1}")
    
    return {'wt': wt, 'dt': dt}

if __name__ == '__main__':
    rawdocs = [
        'lokasi curug indah sejuk baik kondisi jalan sempit becek hujan',
        'saran hujan jalan curug licin becek ojek antar curug',
        'tempat bagus akses curug jalan tanah hujan anjur',
        # 'eat turkey on turkey day holiday',
        # 'i like to eat cake on holiday',
        # 'turkey trot race on thanksgiving holiday',
        # 'snail race the turtle',
        # 'time travel space race',
        # 'movie on thanksgiving',
        # 'movie at air and space museum is cool movie',
        # 'aspiring movie star',
        # 'akses mudah tempat indah pisan seperti curug',
        # 'ada fasilitas outbound paintball pegawai cukup ramah',
        # 'tempat enak buat hiking harga makanan cukup terjangkau',
        # 'tempat bagus cocok buat healing keluarga banyak spot foto',
        # 'bagus untuk camping dan melihat sunrise tarif relatif murah',
    ]

    docs = [doc.split(' ') for doc in rawdocs]
    vocab = list(dict.fromkeys(chain.from_iterable(docs)))

    lda_model = LDA1(docs=docs, vocab=vocab, K=2, alpha=1, eta=1, iterations=1)

    wt = lda_model['wt']
    dt = lda_model['dt']
    # print(lda_model)

    # print(sum(lda_model['wt'][0]))

    K = 2
    eta = 1
    alpha = 1
    # print(wt)
    phi = ( wt[0][0] + eta ) / ( sum(wt[0]) + len(vocab) * eta )
    theta = ( dt[0][1] + alpha ) / ( sum(dt[0]) + K * alpha )

    phi_new = []
    for k in range(K):
        # print(wt[k])
        left_new = []
        for i in range(len(wt[k])):
            phi = ( wt[0][i] + eta ) / ( sum(wt[0]) + len(vocab) * eta )
            left_new.append(round(phi, 5))
            # print(wt[k])
        phi_new.append(left_new)
    
    print(phi_new)

    theta_new = []
    for i in range(len(dt)):
        # print(wt[k])
        right_new = []
        for k in range(K):
            theta = ( dt[0][k] + eta ) / ( sum(dt[i]) + len(vocab) * eta )
            right_new.append(round(theta, 5))
            # print(wt[k])
        theta_new.append(right_new)
    
    print(theta_new)

    # print({'phi': phi, 'theta': theta})

    print(round(phi_new[0][0] * theta_new[0][0], 5))
    print(round(phi_new[1][0] * theta_new[0][1], 5))

    # print(phi)