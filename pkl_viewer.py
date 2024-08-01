import pickle

ls = ['1034']
for fn in ls:
    with open(fn + '.pkl', 'rb') as f:
        data = pickle.load(f)
    print(fn)
    # print(data._node)
    # print(data.edges)
    # print(type(data))
    # 遍历所有节点
    for node in data.nodes():
        print(node)

    # 遍历所有边
    for edge in data.edges():
        print(edge)

    print(data.nodes[0])
    # print(data.nodes[1])
    # print(data.nodes[2])
    # print(data.edges[0, 1])
    # print(data.edges[0, 2])
