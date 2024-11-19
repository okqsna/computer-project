"""check a graph for a hamiltonian cycle"""

def check_for_ham(graph: list[tuple]) -> list | str:
    """
    Checks a graph if it has a hamiltonian cycle or not

    :param graph: list[tuple], a list of tuples, where the first number
    is the vertice, and the second is the connection

    :return: returns either a list of the vertices of the hamiltonian
    cycle, or says that it does nto exist

    >>> g = [(1,2),(2,3),(3,1)]
    >>> check_for_ham(g)
    [1, 2, 3, 1]
    >>> g1 = [(1, 4), (4, 3), (3, 5), (5, 1)]
    >>> check_for_ham(g1)
    [1, 4, 3, 5, 1]
    >>> g2 = [(1, 5), (5, 4), (4, 3), (3, 2), (2, 1)]
    >>> check_for_ham(g2)
    [1, 5, 4, 3, 2, 1]
    >>> g3 = [(1, 2), (2, 4), (5, 1)]
    >>> check_for_ham(g3)
    'There is not hamiltonian cycle for this graph'
    >>> g4 = [(1, 2), (2, 3), (3, 2), (2, 1)]
    >>> check_for_ham(g4)
    'There is not hamiltonian cycle for this graph'
    >>> g4 = [(1, 2), (2, 3), (3, 2), (2, 1)]
    """

    if len(graph) < 3:
        return "To have a hamiltonian graph you need at least 3 vertices!"

    con_dic = {}

    for tup in graph:
        # make connections for first vertices
        if tup[0] in con_dic:
            con_dic[tup[0]].append(tup[1])
        elif tup[0] not in con_dic:
            con_dic[tup[0]] = [tup[1]]
        # Give second vertice con to first
        if tup[1] in con_dic:
            con_dic[tup[1]].append(tup[0])
        elif tup[1] not in con_dic:
            con_dic[tup[1]] = [tup[0]]

    # Check dic for duplicates
    for key1, val1 in con_dic.items():
        con_dic[key1] = list(set(val1))

    # Start extremely cool checking logic (please help me)
    all_keys = [k for k in con_dic]

    # Make all possible
    def permute(nodes):

        if len(nodes) == 1:
            return [nodes]

        perm = []
        n_len = len(nodes)

        for i in range(n_len):
            current = nodes[i]
            remaining = nodes[:i] + nodes[i+1:]

            for p in permute(remaining):
                perm.append([current] + p)

        return perm

    all_possible = permute(all_keys)


    # Add the first to the end of each list
    for a in all_possible:
        a.append(a[0])


    # Check which of all are good
    for path in all_possible:
        test_len = len(path)
        test = 0
        for index , z in enumerate(path):
            if index == 0:
                test += 1
                continue
            elif z in con_dic[path[index - 1]]:
                test += 1
            else:
                break

        if test == test_len:
            return path

    return "There is not hamiltonian cycle for this graph"

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())
