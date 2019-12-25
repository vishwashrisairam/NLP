from tagger import *


def main():
    #pass
    t=Tagger()
    corpus=t.load_corpus('brown_modified')
    t.initialize_counts(corpus)
    # print(corpus[:3])
    S1="The Secretariat is expected to race tomorrow ."
    S2="People continue to enquire the reason for the race for outer space ."
    
    t.sentence=S1
    t.initialize_probabilities(corpus)
    pos_tags=t.viterbi_decode(t.sentence)
    print(t.sentence)
    print(pos_tags)
    
    t.sentence=S2
    t.initialize_probabilities(corpus)
    pos_tags=t.viterbi_decode(t.sentence)
    print(t.sentence)
    print(pos_tags)

if __name__=="__main__":
    main()