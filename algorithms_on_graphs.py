from copy import deepcopy


def convert_to_directed(graph: list[set]) -> list[tuple]:
    """
    Converts undirected graphs into directed.

    >>> graph = [{'a', 'b'}, {'b', 'c'}, {'c', 'a'}]
    >>> set(convert_to_directed(graph)) == \
{('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'a'), ('a', 'c')}
    True
    >>> set(convert_to_directed([{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}])) == \
set([('b', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'b'), ('d', 'c'), ('c', 'd'), ('d', 'a'), ('a','d')])
    True
    """
    output = []
    for edge in graph:
        ver1, ver2 = edge
        output.extend([(ver1, ver2), (ver2, ver1)])

    return output


def euler_cycle(graph: list[tuple | set]) -> list[tuple | set]:
    """
    :param graph: list of tuples of letters that symbolyize verteces.
    :return: list fo all possible euler cycles.

    # >>> graph = [('a', 'b'), ('c', 'b'), ('d', 'c'), ('d', 'a'), \
    #              ('b', 'd'), ('b', 'd')]
    # >>> euler_cycle(graph)
    # ['abdcbd']

    >>> graph1 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}]
    >>> euler_cycle(graph1)
    ['abcd']
    """
    # deepcopy to be on the safe side
    p_graph = deepcopy(graph)

    if flag := isinstance(graph[0], set):
        p_graph = convert_to_directed(p_graph)

    # calculate_way writes all possible cycles here
    all_cycles = []

    def calculate_way(graph: list[tuple], position: str, way: str) -> list[tuple] | str:
        """
        Takes a graph and current position. Check where you can go from your current position
        and creates all these possible ways.
        """
        graph = deepcopy(graph)
        for pair in graph:
            if pair[0] == position:
                if pair[1] == vertex:
                    all_cycles.append(way+vertex)
                graph.remove(pair)
                if flag:
                    graph.remove(pair[::-1])
                calculate_way(graph, pair[1], way+pair[1])

    # We need to decide what will be start vertice
    vertex = 'a'
    calculate_way(p_graph, vertex, vertex)
    edges = [ver1 + ver2 for ver1, ver2 in p_graph]

    output = []


    if flag:
        for way in all_cycles:
            for edge in edges:
                if edge not in way and edge[::-1] not in way:
                    break
            else:
                output.append(way[:-1])

    # Selects those cycles that contain all edges
    else:
        for way in all_cycles:
            for edge in edges:
                if edge not in way:
                    break
            else:
                output.append(way[:-1])

    return output if output else 'There is no euler cycle for this graph'



if __name__ == '__main__':
    import doctest
    doctest.testmod()