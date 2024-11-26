"""Computer project from discrete mathematics"""

from itertools import permutations
from copy import deepcopy

def readfile(file_name: str) -> list[tuple]:
    """
    The function reads the file .dot and makes 
    the list of tuples of neighbour vertices.

    :param file_name: str, the name of the file.
    :return: list[tuple], the list for graph.
    """
    res = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if "-" in line:
                vertices = line.strip().split(" -> ")
                i = 0
                while i != len(vertices) - 1:
                    res.append((int(vertices[i]), int(vertices[i + 1])))
                    i += 1
    return res

def to_undirected(oriented: list[tuple]) -> list[set]:
    """
    The function makes from the list of tuples a list of sets.

    :param oriented: list[tuple], the list for oriented graph.
    :return: list[list], the list for undirected graph.
    >>> to_undirected([(1, 2), (2, 5), (2, 3), (3, 4), (1, 3), (4, 5), (1, 5)])
    [{1, 2}, {2, 5}, {2, 3}, {3, 4}, {1, 3}, {4, 5}, {1, 5}]
    """
    return [set(ver) for ver in oriented]

def convert_to_directed(graph: list[set]) -> list[tuple]:
    """
    Converts undirected graphs into directed.

    >>> graph = [{'a', 'b'}, {'b', 'c'}, {'c', 'a'}]
    >>> sorted(convert_to_directed(graph)) == \
sorted([('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'b'), ('c', 'a'), ('a', 'c')])
    True
    >>> sorted(convert_to_directed([{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}])) == \
sorted([('b', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'b'), ('d', 'c'), ('c', 'd'), \
('d', 'a'), ('a','d')])
    True
    """
    output = []
    for edge in graph:
        ver1, ver2 = edge
        output.extend([(ver1, ver2), (ver2, ver1)])

    return output


def euler_cycle(graph: list[tuple | set]) -> list[list[str]]:
    """
    Check whether 

    :param graph: list of tuples of letters that symbolyize verteces.
    :return: list of all possible euler cycles.

    >>> graph = [('a', 'b'), ('c', 'b'), ('d', 'c'), ('d', 'a'), \
                 ('b', 'd'), ('b', 'd')]
    >>> euler_cycle(graph)
    ['abdcbd']

    >>> graph1 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}]
    >>> euler_cycle(graph1)
    ['abcd', 'adcb']
    >>> graph2 = [{'a', 'b'}, {'b', 'c'}, {'c', 'd'}, {'d', 'a'}, {'a', 'e'}, {'b', 'e'}, \
{'c', 'e'}, {'d', 'e'}, {'f', 'd'}, {'f', 'c'}, {'g', 'a'}, {'g', 'b'}]
    >>> 'agbedabcdfce' in euler_cycle(graph2)
    True
    """
    # deepcopy to be on the safe side
    p_graph = deepcopy(graph)
    length = len(graph)

    if flag := isinstance(graph[0], set):
        p_graph = convert_to_directed(p_graph)

    # calculate_way writes all possible cycles here
    all_cycles = []

    def calculate_way(graph: list[tuple], position: str, way: str, r_p: tuple) -> list[tuple] | str:
        """
        Takes a graph and current position. Check where you can go from your current position
        and creates all these possible ways.
        """
        graph = deepcopy(graph)
        graph.remove(r_p)
        if flag:
            graph.remove(r_p[::-1])
        for pair in graph:
            if pair[0] == position:
                if pair[1] == vertex and len(way) == length:
                    all_cycles.append(way+vertex)
                calculate_way(graph, pair[1], way+pair[1], pair)

    # We need to decide what will be start vertice
    vertex = 'a'
    calculate_way(p_graph + ['*', '*'], vertex, vertex, '*')
    edges = [ver1 + ver2 for ver1, ver2 in graph]

    output = []

    if flag:
        # return all_cycles
        # Selects those cycles that contain all edges for undirected graph
        for cycle in all_cycles:
            for edge in edges:
                # Checks whether this cycle contains this edge
                if edge not in cycle and edge[::-1] not in cycle:
                    break
            else:
                # If this cycle has all edges we add it to output
                output.append(cycle[:-1])

    else:
        # Selects those cycles that contain all edges for directed graph
        for cycle in all_cycles:
            for edge in edges:
                # Checks whether this cycle contains this edge
                if edge not in cycle:
                    break
            else:
                # If this cycle has all edges we add it to output
                output.append(cycle[:-1])

    return sorted(set(output)) if output else 'There is no euler cycle for this graph'


def check_for_ham(graph: list[tuple]) -> list | str:
    """
    Checks a graph if it has a hamiltonian cycle or not

    :param graph: list[tuple], a list of tuples, where the first number
    is the vertice, and the second is the connection

    :return: returns either a list of the vertices of the hamiltonian
    cycle, or says that it doesn't exist

    # UNORIENTED
    >>> g = [{1, 2}, {2, 3}, {3, 1}]
    >>> check_for_ham(g)
    [1, 2, 3, 1]

    >>> g1 = [{1, 4}, {4, 3}, {3, 5}, {5, 1}]
    >>> check_for_ham(g1)
    [1, 4, 3, 5, 1]

    >>> g2 = [{1, 5}, {5, 4}, {4, 3}, {3, 2}, {2, 1}]
    >>> check_for_ham(g2)
    [1, 5, 4, 3, 2, 1]

    >>> g3 = [{1, 2}, {2, 4}, {5, 1}]
    >>> check_for_ham(g3)
    'There is no Hamiltonian cycle for this graph'

    >>> g4 = [{1, 2}, {2, 3}, {3, 2}, {2, 1}]
    >>> check_for_ham(g4)
    'There is no Hamiltonian cycle for this graph'

    # ORIENTED
    >>> g_oriented = [(1, 2), (2, 3), (3, 1)]
    >>> check_for_ham(g_oriented)
    [1, 2, 3, 1]

    >>> g1_oriented = [(1, 4), (4, 3), (3, 5), (5, 1)]
    >>> check_for_ham(g1_oriented)
    [1, 4, 3, 5, 1]

    >>> g2_oriented = [(1, 5), (5, 4), (4, 3), (3, 2), (2, 1)]
    >>> check_for_ham(g2_oriented)
    [1, 5, 4, 3, 2, 1]

    >>> g3_oriented = [(1, 2), (2, 4), (5, 1)]
    >>> check_for_ham(g3_oriented)
    'There is no Hamiltonian cycle for this graph'

    >>> g4_oriented = [(1, 2), (2, 3), (3, 2), (2, 1)]
    >>> check_for_ham(g4_oriented)
    'There is no Hamiltonian cycle for this graph'
    """

    if len(graph) < 3:
        return "To have a Hamiltonian graph you need at least 3 vertices!"

    # Check what type of graph
    unoriented = None

    if isinstance(graph[0], set):
        unoriented = True

    con_dic = {}


    for tup in graph:
        tup = list(tup)
        # make connections for first vertices
        if tup[0] in con_dic:
            con_dic[tup[0]].append(tup[1])
        elif tup[0] not in con_dic:
            con_dic[tup[0]] = [tup[1]]
        if unoriented:
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

    test_len = len(all_possible[0])
    # Check which of all are good
    for path in all_possible:
        test = 1
        for index in range(1, test_len):
            if path[index] in con_dic[path[index - 1]]:
                test += 1
            elif path[index] not in con_dic[path[index - 1]]:
                break

        if test == test_len:
            return path


    return "There is no Hamiltonian cycle for this graph"

def to_matrix(graph_list: list[tuple]) -> list[list]:
    """
    The function makes the matrix for the graph.

    :param graph_list: list[tuple], the graph.
    :return: list[list], matrix.
    >>> to_matrix([(1, 2), (2, 3), (3, 4), (1, 5), (5, 3)])
    [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]
    """
    leng = max(max(i[0], i[1]) for i in graph_list)
    return [[1*((i+1, j+1) in graph_list) for i in range(leng)] for j in range(leng)]

def to_symetric(matrix: list[list]) -> list[list]:
    """
    The function returns the symetric relation.

    :param matrix: list[list], the graph's matrix.
    :return: list[list], the symetric relation.
    >>> to_symetric([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 1], \
[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]])
    [[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0], [1, 0, 1, 0, 0]]
    """
    leng = len(matrix)
    return [[matrix[i][j] | matrix[j][i] for i in range(leng)] for j in range(leng)]

def approp(cur: int, graph: list[list[int]], colours: list[int], colour: int) -> bool:
    """
    The function checks wheather we can assign colour to the vertice.

    :param cur: int, the number of curren vertice.
    :param graph: list[list[int]], the matrix of the graph.
    :param colours: list[int], the list with colours' numbers assigned to each vetice.
    :param colour: int, the colour's number wanted to be assigned.
    :return: bool, if the colour is appropriate - True, else - False.
    """
    for i in range(cur):
        if graph[cur][i] and colour == colours[i]:
            return False
    return True

def colouring(graph: list[list[int]], s: int, colours: list[int], cur: int) -> bool | list[int]:
    """
    The function is "colouring" the graph.

    :param graph: list[list[int]], the matrix of the graph.
    :param s: int, the number of vetices.
    :param colours: list[int], the list with colours' numbers assigned to each vetice.
    :param cur: int, the number of curren vertice.
    :return: bool(False if we can't colour the graph) or the chenged list colours.
    """
    if cur == s:
        return True

    for i in range(1, s+1):
        if approp(cur, graph, colours, i):
            colours[cur] = i

            if colouring(graph, s, colours, cur+1):
                return colours

            colours[cur] = 0
    return False

def get_colour_seq(file_name: str, colour_list: list) -> str:
    """
    The function returns the result of colouring and the text if not possible.

    :param file_name: str, the name of .dot file.
    :return: str, the chosen numbers (only 3).
    :return: str, the colour sequence in string with whight fpaces.
    """
    matrix = to_symetric(to_matrix(readfile(file_name)))
    SIZE = len(matrix)
    chosen_colours = [0]*SIZE
    if colouring(matrix, SIZE, chosen_colours, 0):
        chosen_colours = colouring(matrix, SIZE, chosen_colours, 0)
        result = " ".join(map(lambda x: colour_list[x-1], chosen_colours))
    else:
        result = "The coloring is imposible."
    return result

def write_colour(file_in: str, file_out: str, colours: list[str]) -> None:
    """
    The function writes the coloured graph into the file.

    :param file_in: str, the name of input .dot file.
    :param file_out: str, the name of output .dot file.
    :param colours: str, the chosen numbers (only 3).
    :return: None
    """
    with open(file_out, "w+", encoding='utf-8') as file, \
    open(file_in, 'r', encoding='utf-8') as f:
        sequence = get_colour_seq(file_in, colours)
        num = 0
        for line in f.readlines()[:-1]:
            file.write(line)
            for i in line:
                if i.isnumeric():
                    num = max(num, int(i))
        for i in range(num):
            file.write(f'\t{i+1} [shape = circle style = filled color="{sequence.split()[i]}"]\n')
        file.write("}")
=======
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

    # getting values from additional functions
    graph_new = to_oriented(graph)
    neighbour_vertix = get_neighbouring_values(graph_new)

    # choosing a vertex to start with, setting its color
    start_vertix = list(neighbour_vertix)[0]
    color_vertices = {start_vertix: "green"}

    visited_vertices = set()
    not_visited_vertices = [start_vertix]

    while not_visited_vertices:
        # setting vertix to work with
        current_vertix = not_visited_vertices[0]
        if current_vertix not in visited_vertices:
            # visiting vertix
            visited_vertices.add(current_vertix)
            # setting color for neighbours
            color_vertix = "black" if color_vertices[current_vertix] == "green" else "green"
            for neighbour in neighbour_vertix[current_vertix]:
                if neighbour not in visited_vertices:
                    not_visited_vertices.append(neighbour)
                    color_vertices[neighbour] = color_vertix
                elif color_vertices[current_vertix] == color_vertices[neighbour]:
                    return False
        not_visited_vertices.remove(current_vertix)

    return True


def if_graphs_are_isomorphic(graph_1: list[tuple], graph_2: list[tuple]) -> bool:
    """
    Checks if two graphs are isomorphic

    :param graph_1: a list of edges for the first graph.
    :param graph_2: a list of edges for the second graph.
    :return: bool, True if the graphs are isomorphic, False if not.

    >>> if_graphs_are_isomorphic([(1, 2), (2, 1), (3, 4)], [(2, 1), (1, 2), (4, 3)])
    True
    >>> if_graphs_are_isomorphic([(1, 2), (2, 3)], [(3, 2), (2, 1)])
    False
    """

    def if_graph_is_directed(graph: list[tuple]) -> bool:
        """
        Checks if graph is directed
    
        :param graph: a list of edges, where each edge is a tuple (start, end) representing an undirected edge between 'start' and 'end'.
        :return: bool, True if the graph is directed, False if it is undirected.
    
        >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (4, 5)])
        True
        >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (1, 3)])
        False
        """
        for edge in graph:
            if (edge[1], edge[0]) in graph:
                return False
        return True


    def if_graph_is_undirected(graph: list[tuple]) -> bool:
        """
        Checks if graph is undirected
    
        :param graph: a list of edges, where each edge is a tuple (start, end) representing an undirected edge between 'start' and 'end'.
        :return: bool, True if the graph is undirected, False if it is directed.
    
        >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (4, 5)])
        False
        >>> if_graph_is_directed([(1, 2), (2, 3), (3, 1), (1, 3)])
        True
        """
        for edge in graph:
            if (edge[1], edge[0]) in graph:
                return True
        return False

    def directed_isomorphism(graph_1: list[tuple], graph_2: list[tuple]) -> bool:
        """Checks if directed graphs are isomorphic
    
        :param graph_1: a list of directed edges for the first graph.
        :param graph_2: a list of directed edges for the second graph.
        :return: bool, True if the graphs are isomorphic, False if not.
    
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

        for perm in permutations(vertices_2):
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
    
        :param graph_1: a list of undirected edges for the first graph.
        :param graph_2: a list of undirected edges for the second graph.
        :return: bool, True if the graphs are isomorphic, False if not.
    
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

        for perm in permutations(vertices_2):
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

def write_results(graph_one: list[tuple], graph_two: list[tuple], file_path: str) -> None:
    """
    Function writes the result of all performed 
    algorithms to a given file.

    :param graph_one: list[tuple], given graph
    :param graph_two: list[tuple], given graph (for isomorphic check case)
    :param file_path: str, Name of file to write result to.

    """
    euler_cycle_res = euler_cycle(graph_one)
    hamiltonian_res = check_for_ham(graph_one)
    bipartite_res = bipartite_graph_check(graph_one)
    isomorphic_res = if_graphs_are_isomorphic(graph_one, graph_two)


    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f'All possible Euler cycles: {euler_cycle_res}')
        file.write(f'Existence of Hamiltonian cycle: {hamiltonian_res}')
        file.write(f'Graph is bipartite: {bipartite_res}')
        file.write(f'Graph is isomporphic: {isomorphic_res}')

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

# write_colour("graph.dot", "colored.dot", ["red", "green", "blue"])