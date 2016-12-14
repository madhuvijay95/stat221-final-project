import sys

with open(sys.path[0] + '\\ap.txt', 'r') as f:
    corpus = [line[:-1] for line in f.readlines()]
start_lines = filter(lambda ind : corpus[ind]=='<TEXT>', range(len(corpus)))
end_lines = filter(lambda ind : corpus[ind]==' </TEXT>', range(len(corpus)))
# check that every </text> follows exactly 2 lines after the corresponding <text>
assert (reduce(lambda a,b : a and b, map(lambda tup : tup[1]-tup[0]==2, zip(start_lines, end_lines))))

doc_indices = map(lambda x : x+1, start_lines)
corpus = [corpus[ind]+'\n' for ind in doc_indices]

with open(sys.path[0] + '\\ap_clean.txt', 'w') as f:
    f.writelines(corpus)
