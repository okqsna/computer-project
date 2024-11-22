"""
Module checks whether given graph is bipartite.
"""

def to_oriented(graph: list[tuple]) -> list[tuple]:
    """
    Function transforms not oriented graph into oriented.

    :param graph: list[tuple], given graph
    :return: list[tuple], updated graph
    
    >>> graph = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    >>> to_oriented(graph)
    [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (2, 1), (3, 1), (4, 2), (4, 3), (5, 4)]
    """
    graph_new = graph
    for v1, v2 in graph:
        if (v2, v1) not in graph:
            graph_new.append((v2, v1))
    return graph_new

def get_neighbouring_values(graph: list[tuple]) -> dict:
    """
    Function looks for every neighbouring vertix 
    and returns it as a dictionary.

    :param graph: given graph
    :return: dict, key is a vertex, value is all of the neighbouring vertices

    >>> graph = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (2, 1), (3, 1), (4, 2), (4, 3), (5, 4)]
    >>> get_neighbouring_values(graph)
    {1: {2, 3}, 2: {1, 4}, 3: {1, 4}, 4: {2, 3, 5}, 5: {4}}
    """
    neighbour_vertex = {}
    for v1, v2 in graph:
        if v1 not in neighbour_vertex:
            neighbour_vertex.setdefault(v1, set())
        if v2 not in neighbour_vertex:
            neighbour_vertex.setdefault(v2, set())

        neighbour_vertex[v1].add(v2)
        neighbour_vertex[v2].add(v1)
    return neighbour_vertex


def bipartite_graph_check(graph: list[tuple])-> bool:
    """
    Function checks if a given graph is bipartite.

    A graph is bipartite if its vertices can be divided into two disjoint sets 
    such that no two vertices within the same set are adjacent.
    
    Using a breadth-first search (BFS) approach 
    to check whether the graph satisfies this property
    by coloring vertices into 2 colors.

    :param graph: list[tuple], given graph
    :return: bool, function returns True if graph is bipartite, 
    returns False when it is not.

    >>> graph = [(1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 4)] 
    >>> bipartite_graph_check(graph)
    True
    >>> graph = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5)]
    >>> bipartite_graph_check(graph)
    False
    """

    # getting values from additional functions
    graph_new = to_oriented(graph)
    neighbour_vertix = get_neighbouring_values(graph_new)

    # choosing a vertex to start with, setting its color
    start_vertix = list(neighbour_vertix)[0]
    color_vertices = {start_vertix: "green"}

    visited_vertices = set()
    not_visited_vertices = [start_vertix]

    while not_visited_vertices:
        current_vertix = not_visited_vertices[0]
        if current_vertix not in visited_vertices:
            visited_vertices.add(current_vertix)
            color_vertix = "black" if color_vertices[current_vertix] == "green" else "green"
            for neighbour in neighbour_vertix[current_vertix]:
                if neighbour not in visited_vertices:
                    not_visited_vertices.append(neighbour)
                    color_vertices[neighbour] = color_vertix
                elif color_vertices[current_vertix] == color_vertices[neighbour]:
                    return False
        not_visited_vertices.remove(current_vertix)

    return True

if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
