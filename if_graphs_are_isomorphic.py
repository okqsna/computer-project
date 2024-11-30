"""Checks if graphs are isomorphic""" 

def if_graph_is_directed(graph: list[tuple]) -> bool:
    """
    Checks if graph is directed

    Args:
        graph (list[tuple]): A list of edges, where each edge is a tuple (start, end) representing a directed edge from 'start' to 'end'.

    Returns:
        bool: True if the graph is directed, False if it is undirected.

    >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (4, 5)])
    True
    >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (1, 3)])
    False
    """
    for edges in graph:
        for edge in edges:
            if (edge[1], edge[0]) in graph:
                return False
    return True


def if_graph_is_undirected(graph: list[tuple]) -> bool:
    """
    Checks if graph is undirected

    Args:
        graph (list[tuple]): A list of edges, where each edge is a tuple (start, end) representing an undirected edge between 'start' and 'end'.

    Returns:
        bool: True if the graph is undirected, False if it is directed.

    >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (4, 5)])
    False
    >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (1, 3)])
    True
    """
    for edges in graph:
        for edge in edges:
            if (edge[1], edge[0]) in graph:
                return True
    return False

def directed_isomorphism(graph_1: list[tuple], graph_2: list[tuple]) -> bool:
    """Checks if directed graphs are isomorphic

    Args:
        graph_1 (list[tuple]): A list of directed edges for the first graph.
        graph_2 (list[tuple]):A list of directed edges for the second graph.

    Returns:
        bool: True if the graphs are isomorphic, False if not.

     >>> directed_isomorphism([(1, 2), (2, 3), (3, 1)], [(3, 2), (2, 1), (1, 3)])
    True
    >>> directed_isomorphism([(1, 2), (2, 3), (3, 4)], [(4, 3), (3, 2), (2, 1)])
    False
    """
    if len(graph_1) != len(graph_2):
        return False

    vertices_1 = set()
    vertices_2 = set()

    for start, end in graph_1:
        vertices_1.add(start)
        vertices_1.add(end)
    for start, end in graph_2:
        vertices_2.add(start)
        vertices_2.add(end)

    if len(vertices_1) != len(vertices_2):
        return False

    for perm in permute(vertices_2):
        vertex_match = {}
        for u, v in enumerate(vertices_1):
            vertex_match[v] = perm[u]

        new_graph = []
        for start, end in graph_1:
            new_graph.append((vertex_match[start], vertex_match[end]))

        if sorted(new_graph) == sorted(graph_2):
            return True

    return False


def undirected_isomorphism(graph_1: list[set], graph_2: list[set]) -> bool:
    """
    Checks if undirected graphs are isomorphic

    Args:
        graph_1 (list[tuple]): A list of undirected edges for the first graph.
        graph_2 (list[tuple]):A list of undirected edges for the second graph.

    Returns:
        bool: True if the graphs are isomorphic, False if not.

    >>> undirected_isomorphism([{1, 2}, {2, 3}, {3, 4}], [{4, 3}, {2, 1}, {3, 2}])
    True
    >>> undirected_isomorphism([{1, 2}, {3, 4}], [{1, 2}, {3, 5}])
    False
    """
    if len(graph_1) != len(graph_2):
        return False

    vertices_1 = set()
    vertices_2 = set()

    for start, end in graph_1:
        vertices_1.add(start)
        vertices_1.add(end)
    for start, end in graph_2:
        vertices_2.add(start)
        vertices_2.add(end)

    if len(vertices_1) != len(vertices_2):
        return False

    for perm in permute(vertices_2):
        vertex_match = {}
        for u, v in enumerate(vertices_1):
            vertex_match[v] = perm[u]

        new_graph = []
        for start, end in graph_1:
            new_graph.append((vertex_match[start], vertex_match[end]))
            new_graph.append((vertex_match[end], vertex_match[start]))

        if sorted(new_graph) == sorted(graph_2):
            return True

    return False

def if_graphs_are_isomorphic(graph_1: list[tuple], graph_2: list[tuple]) -> bool:
    """
    Checks if two graphs are isomorphic

    Args:
        graph_1 (list[tuple]): A list of edges for the first graph.
        graph_2 (list[tuple]): A list of edges for the second graph.

    Returns:
        bool: True if the graphs are isomorphic, False if not.

    >>> if_graphs_are_isomorphic([(1, 2), (2, 1), (3, 4)], [(2, 1), (1, 2), (4, 3)])
    True
    >>> if_graphs_are_isomorphic([(1, 2), (2, 3)], [(3, 2), (2, 1)])
    False
    """
    directed_graph_1 = if_graph_is_directed(graph_1)
    directed_graph_2 = if_graph_is_directed(graph_2)
    undirected_graph_1 = if_graph_is_undirected(graph_1)
    undirected_graph_2 = if_graph_is_undirected(graph_2)

    if directed_graph_1 != directed_graph_2:
        return False
    if undirected_graph_1 != undirected_graph_2:
        return False

    if directed_graph_1 == directed_graph_2:
        return directed_isomorphism(graph_1, graph_2)
    if undirected_graph_1 == undirected_graph_2:
        return undirected_isomorphism(graph_1, graph_2)


if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
