from copy import deepcopy
def euler_cycle(graph: list[tuple | set]) -> list[list[str]]:
    """
    Check whether 

    :param graph: list[tuple] or list[set], list of tuples of letters that symbolyize verteces.
    :return: list[list[str]],  list of all possible euler cycles.

    >>> graph = [('a', 'b'), ('c', 'b'), ('d', 'c'), ('d', 'a'), \
                 ('b', 'd'), ('b', 'd')]
    >>> euler_cycle(graph)
    "This graph isn't strongly connected"

    >>> graph1 = [('a', 'b'), ('b', 'a'), ('c', 'd'), ('d', 'c'), \
                 ('a', 'd'), ('d', 'a'), ('b', 'd'), ('d', 'b')]
    >>> euler_cycle(graph1)
    'There is no euler cycle for this graph'

    >>> graph2 = [('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'a'), ('a', 'c'), \
                 ('c', 'd'), ('d', 'c'), ('d', 'e'), ('e', 'd'), ('e', 'c'), ('c', 'e')]
    >>> euler_cycle(graph2)
    [['a', 'b', 'c', 'd', 'e', 'c'], ['a', 'b', 'c', 'e', 'd', 'c']]

    >>> graph3 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}]
    >>> euler_cycle(graph3)
    [['a', 'b', 'c', 'd']]

    >>> graph4 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}, {'a', 'e'}, {'b', 'e'}, \
{'c', 'e'}, {'d', 'e'}, {'f', 'd'}, {'f', 'c'}, {'g', 'a'}, {'g', 'b'}]
    >>> ['a', 'e', 'd', 'f', 'c', 'e', 'b', 'c', 'd', 'a', 'b', 'g'] in euler_cycle(graph4)
    True
    """
    if not graph:
        return "This graph is empty"

    if isinstance(graph[0], set):
        length = len(graph)
        p_graph = []
        for edge in graph:
            ver1, ver2 = edge
            p_graph.extend([(ver1, ver2), (ver2, ver1)])
    else:
        p_graph = deepcopy(graph)
        for ver1, ver2 in graph:
            if (ver2, ver1) not in graph:
                return "This graph isn't strongly connected"
        length = len(graph) / 2

    # calculate_way writes all possible cycles here
    all_cycles = []

    def calculate_way(graph: list[tuple], position: str, way: str, r_p: tuple) -> list[tuple] | str:
        """
        Takes a graph and current position. Check where you can go from your current position
        and creates all these possible ways.
        """
        graph = deepcopy(graph)
        graph.remove(r_p)
        graph.remove(r_p[::-1])
        for pair in graph:
            if pair[0] == position:
                if pair[1] == vertex and len(way) == length:
                    all_cycles.append(way)
                calculate_way(graph, pair[1], way+pair[1], pair)

    # This is a vertix from which we move
    vertex = p_graph[0][0] if p_graph[0][0] < p_graph[0][1] else p_graph[0][1]
    calculate_way(p_graph + ['*', '*'], vertex, vertex, '*')

    output = []
    for cycle in all_cycles:
        if 'a' + cycle[1:][::-1] not in output and cycle not in output:
            output.append(cycle)

    return [list(el) for el in sorted(output)] if output else 'There is no euler cycle for this graph'


print(euler_cycle([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'a'), ('a', 'c'), ('c', 'd'), ('d', 'c'), ('d', 'e'), ('e', 'd'), ('e', 'c'), ('c', 'e')]))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
